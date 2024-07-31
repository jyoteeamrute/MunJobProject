# MunJobProject/main.py

import streamlit as st
from streamlit_option_menu import option_menu
from py2neo import Graph
from managers import (
    DatabaseManager,
    StreamlitSkillManager,
    StreamlitCourseManager,
    StreamlitProfessionManager,
    StreamlitPersonManager,
    EmbeddingManager
)
from constants import MENU_ITEMS
from config import Config

# Initialize the DatabaseManager
database_manager = DatabaseManager()

# Initialize the EmbeddingManager
embedding_manager = EmbeddingManager()

# Initialize the Graph database connection
graph = Graph(Config.NEO4J_URI, auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD))

with st.sidebar:
    language_switch = st.radio("Select Language", ("English", "Finnish"))
    labels = MENU_ITEMS[language_switch]
    menu_options = [labels["skills"], labels["courses"], labels["professions"], labels["persons"]]
    menu_icons = ["hammer", "mortarboard", "person-workspace", "person"]
    selected_option = option_menu("MunJob", menu_options, icons=menu_icons, menu_icon="cast", default_index=0)

streamlit_skill_manager = StreamlitSkillManager(database_manager.skill_manager, database_manager, database_manager.embedding_manager)
streamlit_course_manager = StreamlitCourseManager(database_manager.course_manager, database_manager, embedding_manager)
streamlit_profession_manager = StreamlitProfessionManager(database_manager.profession_manager, database_manager, embedding_manager)
streamlit_person_manager = StreamlitPersonManager(database_manager.person_manager, database_manager)

if selected_option == labels["skills"]:
    streamlit_skill_manager.manage_skill(language_switch)
elif selected_option == labels["courses"]:
    streamlit_course_manager.manage_courses(language_switch)
elif selected_option == labels["professions"]:
    streamlit_profession_manager.manage_professions(language_switch)
elif selected_option == labels["persons"]:
    streamlit_person_manager.manage_persons(language_switch)
