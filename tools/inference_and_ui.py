import logging
import os

import streamlit as st

from recsys.config import settings
from recsys.ui.feature_group_updater import get_fg_updater
from recsys.ui.interaction_tracker import get_tracker
from recsys.ui.recommenders import customer_recommendations, llm_recommendations
from recsys.ui.utils import get_deployments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CUSTOMER_IDS = [
    "9e619265e3ae0d2ef96a71577c4aff3474bfa7dd0d60486b42bc8f921c3387c0",
    "a1f7201399574e78b0a1575c50e3b68d116f84e24c0f70c957083da99db6ab5f",
    "19fa659096de20f0c022b9727779e849813ccc82952b3d56e212ab18fa2c0bf3",
    "d9448c8585f1678937deb5118d95b09bf6f41fe00a65b1fb82c7d176c6bfc532",
    "b41d990c8a127dac386dd6c9f2a6ec4ac41185cd21ef2df0a952a8cbdf61ed5d",
]


def initialize_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.title("👒 Fashion Items Recommender")
    st.sidebar.title("⚙️ Configuration")


def initialize_services():
    """Initialize tracker, updater, and deployments"""
    tracker = get_tracker()
    fg_updater = get_fg_updater()

    logger.info("Initializing deployments...")
    with st.sidebar:
        with st.spinner("🚀 Starting Deployments..."):
            articles_fv, ranking_deployment, query_model_deployment = get_deployments()
        st.success("✅ Deployments Ready")

        # Stop deployments button
        if st.button(
            "⏹️ Stop Deployments", key="stop_deployments_button", type="secondary"
        ):
            ranking_deployment.stop()
            query_model_deployment.stop()
            st.success("Deployments stopped successfully!")

    return tracker, fg_updater, articles_fv, ranking_deployment, query_model_deployment


def show_interaction_dashboard(tracker, fg_updater, page_selection):
    """Display interaction data and controls"""
    with st.sidebar.expander("📊 Interaction Dashboard", expanded=True):
        if page_selection == "LLM Recommendations":
            api_key = (
                settings.OPENAI_API_KEY.get_secret_value()
                if settings.OPENAI_API_KEY
                and settings.OPENAI_API_KEY.get_secret_value()
                else None
            )
            if not api_key:
                api_key = st.text_input(
                    "🔑 OpenAI API Key:", type="password", key="openai_api_key"
                )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                st.warning("⚠️ Please enter OpenAI API Key for LLM Recommendations")
            st.divider()

        interaction_data = tracker.get_interactions_data()

        col1, col2, col3 = st.columns(3)
        total = len(interaction_data)
        clicks = len(interaction_data[interaction_data["interaction_score"] == 1])
        purchases = len(interaction_data[interaction_data["interaction_score"] == 2])

        col1.metric("Total", total)
        col2.metric("Clicks", clicks)
        col3.metric("Purchases", purchases)

        st.dataframe(interaction_data, hide_index=True)
        fg_updater.process_interactions(tracker, force=True)


def handle_llm_page(articles_fv, customer_id):
    """Handle LLM recommendations page"""
    if "OPENAI_API_KEY" in os.environ:
        llm_recommendations(articles_fv, os.environ["OPENAI_API_KEY"], customer_id)
    else:
        st.warning("Please provide your OpenAI API Key in the Interaction Dashboard")


def process_pending_interactions(tracker, fg_updater):
    """Process interactions immediately"""
    fg_updater.process_interactions(tracker, force=True)


def main():
    # Initialize page
    initialize_page()

    # Initialize services
    tracker, fg_updater, articles_fv, ranking_deployment, query_model_deployment = (
        initialize_services()
    )

    # Select customer
    customer_id = st.sidebar.selectbox(
        "👤 Select Customer:", CUSTOMER_IDS, key="selected_customer"
    )

    # Page selection
    page_options = ["Customer Recommendations", "LLM Recommendations"]
    page_selection = st.sidebar.radio("📑 Choose Page:", page_options)

    # Process any pending interactions with notification
    process_pending_interactions(tracker, fg_updater)

    # Interaction dashboard with OpenAI API key field
    show_interaction_dashboard(tracker, fg_updater, page_selection)

    # Handle page content
    if page_selection == "Customer Recommendations":
        customer_recommendations(
            articles_fv, ranking_deployment, query_model_deployment, customer_id
        )
    else:  # LLM Recommendations
        handle_llm_page(articles_fv, customer_id)


if __name__ == "__main__":
    main()
