"""Streamlit UI: profile, parallel diet/fitness plans, and streaming follow-up Q&A."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from chains import build_diet_chain, build_fitness_chain, build_qa_chain
from config import get_settings
from llm_factory import get_chat_model
from prompts import format_user_profile

DISCLAIMER = (
    "This app is for **informational purposes only**. It is not medical, nutrition, or "
    "training advice and not a substitute for a qualified professional."
)

ACTIVITY_LEVELS = ["Sedentary", "Light", "Moderate", "Active", "Very active"]
DIETARY_PREFS = [
    "Omnivore",
    "Vegetarian",
    "Vegan",
    "Keto",
    "Low carb",
    "Mediterranean",
    "Other",
]


def _init_session_state() -> None:
    defaults = {
        "diet_plan": None,
        "fitness_plan": None,
        "chat_messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _format_chat_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No prior messages in this session."
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines) if lines else "No prior messages in this session."


def _profile_payload(
    age: int,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
    dietary_preference: str,
    fitness_goal: str,
    constraints: str,
) -> dict:
    c = constraints.strip() or "None specified."
    return {
        "age": age,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "activity_level": activity_level,
        "dietary_preference": dietary_preference,
        "fitness_goal": fitness_goal.strip(),
        "constraints": c,
    }


def _profile_complete(
    fitness_goal: str,
) -> bool:
    return bool(fitness_goal and fitness_goal.strip())


def _try_build_llm(
    provider: str,
    model_name: str,
    temperature: float,
) -> tuple[BaseChatModel | None, str | None]:
    try:
        llm = get_chat_model(provider, model_name.strip(), temperature=temperature)
        return llm, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Could not create model: {e}"


def main() -> None:
    st.set_page_config(
        page_title="AI Health & Fitness Planner",
        layout="wide",
    )
    _init_session_state()
    settings = get_settings()

    st.title("AI Health & Fitness Planner")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Model")
        provider = st.selectbox(
            "Provider",
            options=["ollama", "gemini"],
            index=0 if settings.default_llm_provider.lower() != "gemini" else 1,
        )
        default_model = (
            settings.default_gemini_model
            if provider == "gemini"
            else settings.default_ollama_model
        )
        model_name = st.text_input("Model name", value=default_model)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)

        if st.button("Test connection"):
            llm, err = _try_build_llm(provider, model_name, temperature)
            if err:
                st.error(err)
            else:
                try:
                    assert llm is not None
                    llm.invoke([HumanMessage(content="Reply with the single word: ok")])
                    st.success("Connection OK (model responded).")
                except Exception as e:
                    st.error(f"Model created but call failed: {e}")

    col_main, col_chat = st.columns([1, 1], gap="large")

    with col_main:
        st.subheader("Your profile")
        age = st.number_input("Age", min_value=10, max_value=120, value=30, step=1)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        activity_level = st.selectbox("Activity level", options=ACTIVITY_LEVELS, index=2)
        dietary_preference = st.selectbox("Dietary preference", options=DIETARY_PREFS, index=0)
        fitness_goal = st.text_input("Fitness goal", placeholder="e.g. lose fat, run 5K, build strength")
        constraints = st.text_area(
            "Optional constraints",
            placeholder="Allergies, injuries, equipment, time limits, foods to avoid…",
            height=80,
        )

        profile_ok = _profile_complete(fitness_goal)
        if not profile_ok:
            st.warning("Enter a **fitness goal** to enable plan generation.")

        gen_disabled = not profile_ok or not model_name.strip()
        if st.button("Generate personalized plans", type="primary", disabled=gen_disabled):
            payload = _profile_payload(
                age,
                weight_kg,
                height_cm,
                activity_level,
                dietary_preference,
                fitness_goal,
                constraints,
            )
            llm_d, err_d = _try_build_llm(provider, model_name, temperature)
            llm_f, err_f = _try_build_llm(provider, model_name, temperature)
            if err_d or err_f:
                st.error(err_d or err_f)
            else:
                assert llm_d is not None and llm_f is not None
                diet_chain = build_diet_chain(llm_d)
                fitness_chain = build_fitness_chain(llm_f)
                try:
                    with st.spinner("Generating diet and fitness plans…"):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            f_diet = pool.submit(diet_chain.invoke, payload)
                            f_fit = pool.submit(fitness_chain.invoke, payload)
                            future_kind = {f_diet: "diet", f_fit: "fitness"}
                            errors: list[str] = []
                            diet_text = fitness_text = None
                            for fut in as_completed([f_diet, f_fit]):
                                try:
                                    result = fut.result()
                                except Exception as e:
                                    errors.append(str(e))
                                    continue
                                if future_kind[fut] == "diet":
                                    diet_text = result
                                else:
                                    fitness_text = result
                            if errors:
                                st.error("One or both plan requests failed:\n\n" + "\n".join(errors))
                            if diet_text is not None:
                                st.session_state["diet_plan"] = diet_text
                            if fitness_text is not None:
                                st.session_state["fitness_plan"] = fitness_text
                            if diet_text is not None and fitness_text is not None:
                                st.session_state["chat_messages"] = []
                                st.success("Plans generated. Ask follow-up questions on the right.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        diet_plan = st.session_state.get("diet_plan")
        fitness_plan = st.session_state.get("fitness_plan")
        if diet_plan or fitness_plan:
            tab_diet, tab_fit = st.tabs(["Diet plan", "Fitness plan"])
            with tab_diet:
                st.markdown(diet_plan or "_No diet plan yet._")
            with tab_fit:
                st.markdown(fitness_plan or "_No fitness plan yet._")

    with col_chat:
        st.subheader("Follow-up Q&A")
        if not st.session_state.get("diet_plan") or not st.session_state.get("fitness_plan"):
            st.info("Generate both plans first to unlock chat.")
        else:
            messages = st.session_state["chat_messages"]
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if messages and messages[-1]["role"] == "user":
                pending = messages[-1]["content"]
                history_before = messages[:-1]
                llm_q, err = _try_build_llm(provider, model_name, temperature)
                if err:
                    st.error(err)
                    st.session_state["chat_messages"].pop()
                else:
                    assert llm_q is not None
                    qa_chain = build_qa_chain(llm_q)
                    user_profile = format_user_profile(
                        age=age,
                        weight_kg=weight_kg,
                        height_cm=height_cm,
                        activity_level=activity_level,
                        dietary_preference=dietary_preference,
                        fitness_goal=fitness_goal,
                        constraints=constraints,
                    )
                    qa_input = {
                        "user_profile": user_profile,
                        "diet_plan": st.session_state["diet_plan"],
                        "fitness_plan": st.session_state["fitness_plan"],
                        "chat_history": _format_chat_history(history_before),
                        "question": pending,
                    }
                    acc: list[str] = []
                    with st.chat_message("assistant"):

                        def token_gen():
                            for chunk in qa_chain.stream(qa_input):
                                if chunk:
                                    acc.append(chunk)
                                    yield chunk

                        st.write_stream(token_gen)
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": "".join(acc)}
                    )

            if prompt := st.chat_input("Ask about your plans…"):
                st.session_state["chat_messages"].append(
                    {"role": "user", "content": prompt}
                )
                st.rerun()

        st.divider()
        if st.button("Reset session (plans & chat)", type="secondary"):
            st.session_state["diet_plan"] = None
            st.session_state["fitness_plan"] = None
            st.session_state["chat_messages"] = []
            st.rerun()

    st.divider()
    st.markdown(f"**Disclaimer:** {DISCLAIMER}")


if __name__ == "__main__":
    main()
