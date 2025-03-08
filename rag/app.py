import streamlit as st
import os
import atexit

# Import modules
from database import MongoDB
from auth import show_login_page, show_signup_page, check_session, logout_user
from rag import EnhancedRAG
from notebooks import show_notebooks_page, show_notebook_detail_page, show_document_view_page
from settings import show_settings_page
from chat import show_chat_page
from utils import init_session_state, cleanup_temp_files, set_page_style

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="GPU-Accelerated RAG System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    set_page_style()
    
    # Initialize session state
    init_session_state()
    
    # Initialize MongoDB connection
    if not st.session_state.mongo_db:
        # Set MongoDB connection string from environment variable or default
        connection_string = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        st.session_state.mongo_db = MongoDB(connection_string)
    
    # Check for existing user session
    logged_in = check_session(st.session_state.mongo_db)
    
    # Show authentication page if not logged in
    if not logged_in and st.session_state.user is None:
        # Display app title for non-logged in users
        st.title("🚀 Advanced RAG System with Multiple Modes")
        
        if st.session_state.auth_page == "login":
            show_login_page(st.session_state.mongo_db)
        else:
            show_signup_page(st.session_state.mongo_db)
        return  # Exit early to show only the auth page
    
    # User is logged in, show navigation sidebar
    with st.sidebar:
        st.title("WELCOME ")
        st.markdown(f"Welcome, **{st.session_state.user['name']}**!")
        
        # Navigation options
        st.header("📌 Navigation")
        nav_options = {
            "chat": "💬 Chat",
            "notebooks": "📚 Notebooks",
            "settings": "⚙️ Settings"
        }
        
        selected_nav = st.radio(
            "Go to",
            options=list(nav_options.keys()),
            format_func=lambda x: nav_options[x],
            key="nav_selection",
            index=list(nav_options.keys()).index(st.session_state.page) if st.session_state.page in nav_options else 0
        )
        
        # Update the page if navigation changed
        if selected_nav != st.session_state.page and st.session_state.page in nav_options:
            st.session_state.page = selected_nav
            st.rerun()
        
        # Add logout button
        st.button("Logout", on_click=logout_user, args=(st.session_state.mongo_db,))
    
    # Display different pages based on selection
    if st.session_state.page == "notebooks":
        show_notebooks_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    elif st.session_state.page == "notebook_detail":
        show_notebook_detail_page(
            st.session_state.mongo_db, 
            st.session_state.user['user_id'],
            EnhancedRAG
        )
    elif st.session_state.page == "document_view":
        show_document_view_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    elif st.session_state.page == "settings":
        show_settings_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    else:  # Default to chat page
        show_chat_page(
            st.session_state.mongo_db, 
            st.session_state.user['user_id'],
            EnhancedRAG
        )

if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_temp_files)
    
    # Start the app
    main()