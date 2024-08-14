import re
import streamlit as st
from constants import *


class StreamlitProfessionManager:
    def __init__(self, profession_manager, db_manager, embeddings_manager):
        self.profession_manager = profession_manager
        self.db_manager = db_manager
        self.embedding_manager = embeddings_manager
        self.key_dict = {}

    def add_profession(self, language):
        st.write(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"/>',
            unsafe_allow_html=True)

        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['profession'][language]
        process_switch = st.radio(f"{labels['select']} :", (labels['add'], labels['load_from_file']))

        if process_switch == labels['add']:
            st.subheader(labels["add_profession"])
            profession_source_si = st.text_input(properties["source_si"], key="add_profession_source_si")
            profession_source_id = st.text_input(properties["source_id"], key="add_profession_source_id")
            profession_title_en = st.text_input(f"{properties['title']} *", key="add_profession_title_en")
            profession_title_fi = st.text_input(f"{properties['title_fi']} *", key="add_profession_title_fi")
            profession_description_en = st.text_area(properties["description"], key="add_profession_description_en")
            profession_description_fi = st.text_area(properties["description_fi"], key="add_profession_description_fi")
            profession_skills = st.text_area(properties["skills"], key="add_profession_skills")

            if st.button(key=f'add_profession_button', label=f'€ {labels["add_profession"]}'):
                if not profession_title_en or not profession_title_fi:
                    st.warning(labels["title_empty_warning"])
                else:
                    success = self.profession_manager.add_profession(
                        source_sl=profession_source_si,
                        source_id=profession_source_id,
                        title=profession_title_en,
                        title_fi=profession_title_fi,
                        description=profession_description_en,
                        description_fi=profession_description_fi,
                        skills=profession_skills,
                        warnings_fn=st.warning
                    )

                    if success:
                        st.success(labels["success_add_profession"].format(profession_title=profession_title_en))
                        st.rerun()

        if process_switch == labels['load_from_file']:
            st.subheader(labels["load_from_file"])
            uploaded_file = st.file_uploader(labels["load_from_file"], type="xlsx",
                                             key="load_professions_file_uploader")
            if uploaded_file:
                if st.button(key=f'load_profession_button', label=f'€ {labels["load_from_file"]}'):
                    success = self.profession_manager.load_professions_from_file(uploaded_file, warnings_fn=st.warning)
                    if success:
                        st.success(labels["success_load_professions"])
                        st.rerun()

    def delete_profession(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['profession'][language]
        st.subheader(labels["delete_profession"])

        # Fetch all professions
        all_professions = (self.db_manager.get_all_professions())
        # Check if there are no professions available
        if not all_professions:
            st.warning(labels.get("no_professions_available", "No professions available to delete."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        professions_titles = sorted([profession[language_key] for profession in all_professions])
        cleaned_profession_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in professions_titles]

        # Pair he cleaned titles with the original titles
        profession_pairs = zip(cleaned_profession_titles, professions_titles)

        # Sort based on the cleaned titles
        sorted_profession_pairs = sorted(profession_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_profession_titles = [pair[1] for pair in sorted_profession_pairs]
        selected_profession_title = st.selectbox(properties["title"], sorted_profession_titles,
                                                 key="delete_profession_title_selectbox")

        try:
            # Find the selected course in the list to display its properties
            selected_profession = next(
                profession for profession in all_professions if profession[language_key] == selected_profession_title)

            # Display the properties of the selected course
            for key, display_name in properties.items():
                st.write(f"**{display_name}:** {selected_profession.get(key, 'N/A')}")

            # Button to delete the profession
            if st.button(labels["delete_profession"], key="delete_profession_button"):
                success = self.profession_manager.delete_profession(selected_profession['title'],
                                                                    warnings_fn=st.warning)
                if success:
                    st.success(labels["success_delete_profession"].format(profession_title=selected_profession_title))
                    st.rerun()
        except StopIteration:
            st.warning(f"Profession '{selected_profession_title}' not found.")

    def update_profession(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['profession'][language]
        st.subheader(labels["update_profession"])

        # Fetch all professions
        all_professions = self.profession_manager.get_all_professions()
        # Check if there are no professions available
        if not all_professions:
            st.warning(labels.get("no_professions_available", "No professions available to update."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        professions_titles = sorted ([profession[language_key] for profession in all_professions])
        cleaned_profession_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in professions_titles]

        # Pair he cleaned titles with the original titles
        profession_pairs = zip(cleaned_profession_titles, professions_titles)

        # Sort based on the cleaned titles
        sorted_profession_pairs = sorted(profession_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_profession_titles = [pair[1] for pair in sorted_profession_pairs]

        selected_profession_title = st.selectbox(properties["title"], sorted_profession_titles,
                                                 key="update_profession_title_selectbox")

        try:
            # Find the selected course in the list to display its properties
            selected_profession = next(
                profession for profession in all_professions if profession[language_key] == selected_profession_title)

            for key, display_name in properties.items():
                st.write(f"**{display_name}:** {selected_profession.get(key, 'N/A')}")

            st.subheader(labels["edit_fields"])

            # Display the properties of the selected profession and input fields for new properties, pre-filled with existing data
            new_profession_source_sl = st.text_input(properties['source_si'],
                                                     value=selected_profession.get('source_si', ''),
                                                     key="update_new_profession_source_si")
            new_profession_source_id = st.text_input(properties['source_id'],
                                                     value=selected_profession.get('source_id', ''),
                                                     key="update_new_profession_source_id")
            new_profession_title = st.text_input(f"{properties['title']} *",
                                                 value=selected_profession.get('title', ''),
                                                 key="update_new_profession_title_en")
            new_profession_title_fi = st.text_input(f"{properties['title_fi']} *",
                                                    value=selected_profession.get('title_fi', ''),
                                                    key="update_new_profession_title_fi")

            new_profession_description = st.text_area(properties['description'],
                                                      value=selected_profession.get('description', ''),
                                                      key="update_new_profession_description_en")
            new_profession_description_fi = st.text_area(properties['description_fi'],
                                                         value=selected_profession.get('description_fi', ''),
                                                         key="update_new_profession_description_fi")
            new_profession_skills = st.text_area(properties['skills'], value=selected_profession.get('skills', ''),
                                                 key="update_new_profession_skills")

            # Process the update button click
            if st.button(labels["update_profession"], key="update_profession_button"):

                if not new_profession_title or not new_profession_title_fi:
                    st.warning(labels["title_empty_warning"])
                    return

                updated_properties = {
                    'source_sl': new_profession_source_sl,
                    'source_id': new_profession_source_id,
                    'title_en': new_profession_title,
                    'title_fi': new_profession_title_fi,
                    'description': new_profession_description,
                    'description_fi': new_profession_description_fi,
                    'skills': new_profession_skills,
                }

                success = self.profession_manager.update_profession(selected_profession['title'], **updated_properties)

                if success:
                    st.success(labels["success_update_profession"].format(profession_title=new_profession_title))
                    st.rerun()
        except StopIteration:
            st.warning(f"Profession '{selected_profession_title}' not found.")

    def view_related_nodes(self, language):
        labels = MENU_ITEMS[language]
        st.subheader(labels["view_related_nodes"])

        if 'selected_profession' not in st.session_state:
            st.session_state.selected_profession = None

        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None

        all_professions = self.profession_manager.get_all_professions()

        # Check if there are no professions available
        if not all_professions:
            st.warning(labels.get("no_profession_available", "No professions available to update."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract professions titles for the select box
        professions_titles = sorted([profession[language_key] for profession in all_professions])
        cleaned_profession_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in professions_titles]

        # Pair he cleaned titles with the original titles
        profession_pairs = zip(cleaned_profession_titles, professions_titles)

        # Sort based on the cleaned titles
        sorted_profession_pairs = sorted(profession_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_profession_titles = [pair[1] for pair in sorted_profession_pairs]
        # Select a profession to view related nodes
        selected_profession_title = st.selectbox(labels["select_profession"], sorted_profession_titles,
                                                 key="view_related_nodes_profession_selectbox")

        if selected_profession_title:
            selected_profession = next(profession for profession in all_professions if
                                       profession[language_key] == selected_profession_title)
            st.session_state.selected_profession = selected_profession

            if st.button(labels["view"], key="view_related_to_profession_nodes_button"):
                st.session_state.selected_profession = selected_profession

        if st.session_state.selected_profession:
            # Fetch related nodes
            related_nodes = self.db_manager.get_related_nodes(DEFAULT_PROFESSION_LABEL,
                                                              st.session_state.selected_profession['title'])

            skills = {node_id: node_info for node_id, node_info in related_nodes.items() if
                      DEFAULT_SKILL_LABEL in node_info['labels']}
            new_skills = {node_id: node_info for node_id, node_info in related_nodes.items() if
                          DEFAULT_NEW_SKILL_LABEL in node_info['labels']}

            # Display related new skills
            if new_skills:
                st.subheader(labels["related_new_skills"])
                for node_id, node_info in new_skills.items():
                    st.write(f"**{labels['skill_name']}:** {node_info.get('title', 'N/A')}")
                    st.write(f"**{labels['skill_description']}:** {node_info.get('description', 'N/A')}")
                    st.write(f"**{labels['relationship_type']}:** {node_info.get('relationship_type', 'N/A')}")
                    st.write(f"**{labels['labels']}:** {', '.join(node_info.get('labels', []))}")

                    button_key = f"disconnect_new_skill_{node_id}"
                    if st.button(
                            f"{labels['disconnect']} {node_info.get('title', 'N/A')} {labels['from']} {st.session_state.selected_profession['title']}",
                            key=button_key):
                        st.session_state.selected_node = node_info.get('title', 'N/A')
                        st.session_state.disconnect_button_clicked = button_key

            # Display related skills
            if skills:
                st.subheader(labels["related_skills"])
                for node_id, node_info in skills.items():
                    st.write(f"**{labels['skill_name']}:** {node_info.get('title', 'N/A')}")
                    st.write(f"**{labels['skill_description']}:** {node_info.get('description', 'N/A')}")
                    st.write(f"**{labels['relationship_type']}:** {node_info.get('relationship_type', 'N/A')}")
                    st.write(f"**{labels['labels']}:** {', '.join(node_info.get('labels', []))}")

                    button_key = f"disconnect_skill_{node_id}"
                    if st.button(
                            f"{labels['disconnect']} {node_info.get('title', 'N/A')} {labels['from']} {st.session_state.selected_profession['title']}",
                            key=button_key):
                        st.session_state.selected_node = node_info.get('title', 'N/A')
                        st.session_state.disconnect_button_clicked = button_key

        if 'disconnect_button_clicked' in st.session_state:
            if st.session_state.selected_node:
                # Perform the deletion
                self.db_manager.disconnect_profession_from_skills(st.session_state.selected_profession['title'],
                                                                  [st.session_state.selected_node])
                st.success(
                    f"Disconnected {st.session_state.selected_profession['title']} from {st.session_state.selected_node}")
                st.session_state.selected_node = None
                del st.session_state.disconnect_button_clicked
                st.rerun()

    def connect_to_existing_skills(self, language):
        labels = MENU_ITEMS[language]
        profession_properties = NODE_PROPERTIES['profession'][language]

        # Fetch all professions
        all_professions = self.profession_manager.get_all_professions()

        # Check if there are no courses available
        if not all_professions:
            st.warning(labels.get("no_profession_available", "No professions available to update."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract professions titles for the select box
        professions_titles = sorted([profession[language_key] for profession in all_professions])
        cleaned_profession_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in professions_titles]

        # Pair he cleaned titles with the original titles
        profession_pairs = zip(cleaned_profession_titles, professions_titles)

        # Sort based on the cleaned titles
        sorted_profession_pairs = sorted(profession_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_profession_titles = [pair[1] for pair in sorted_profession_pairs]
        # Select a course to update
        selected_profession_display_title = st.selectbox(labels["select_profession_to_connect_skills"],
                                                         sorted_profession_titles,
                                                         key="connect_to_skills_profession_title_selectbox")

        # Find the selected profession in the list to display its properties
        if selected_profession_display_title:
            selected_profession = next(
                (profession for profession in all_professions if
                 profession[language_key] == selected_profession_display_title), None)

            for key, display_name in profession_properties.items():
                st.write(f"**{display_name}:** {selected_profession.get(key, 'N/A')}")

            if selected_profession:
                try:
                    connected_skills = self.db_manager.get_skills_connected_to_profession(selected_profession['title'])
                    # Remove skills with empty language_key
                    connected_skills = [skill for skill in connected_skills if skill[language_key] != ""]
                    # Extract titles
                    connected_skills_titles = [skill[language_key] for skill in connected_skills]

                    all_skills = self.db_manager.get_all_skills()
                    # Identify unconnected skills
                    unconnected_skills = [skill for skill in all_skills if
                                          skill['title'] not in connected_skills_titles]

                    unconnected_skills = [skill for skill in unconnected_skills if skill[language_key] != ""]

                    unconnected_skills_titles = [skill[language_key] for skill in unconnected_skills]

                    # Calculate similarity of unconnected skills to the course text
                    similar_skills = self.embedding_manager.find_top_similar_skills(
                        selected_profession['title'],
                        selected_profession['description'],
                        NUM_SIMILAR_SKILLS_FOR_PROFESSION_CONNECTION
                    )

                except Exception as e:
                    st.error(f"Error finding similar skills: {e}")
                    return

                # Filter and sort unconnected skills based on similarity
                unconnected_skills_with_similarity = [
                    (skill_title, similarity) for skill_title, similarity in similar_skills if
                    skill_title in unconnected_skills_titles
                ]

                unconnected_skills_with_similarity.sort(key=lambda x: x[1], reverse=True)

                # Prepare the list for multiselect
                formatted_unconnected_skills_titles = [
                    f"{skill_title} s::{similarity:.2f}" for skill_title, similarity in
                    unconnected_skills_with_similarity
                ]

                connected_skills_titles = [skill[language_key] for skill in connected_skills]

                profession_col1, profession_col2, profession_col3 = st.columns([1, 0.8, 1])

                with profession_col1:
                    st.subheader(labels["connected_skills"])
                    selected_connected_skills_titles = [
                        title for title in connected_skills_titles if st.checkbox(title, key=f"connected_{title}")
                    ]

                with profession_col3:
                    st.subheader(labels["unconnected_skills"])
                    selected_unconnected_skills_titles = st.multiselect(" ", formatted_unconnected_skills_titles,
                                                                        key="unconnected_skills_multiselect")

                with profession_col2:
                    if st.button(labels["disconnect_skill"] + ' >>',
                                 key="disconnect_skill_from_profession_button_tab5"):
                        selected_connected_skills = [
                            skill for skill in connected_skills if
                            skill[language_key] in selected_connected_skills_titles
                        ]
                        if selected_connected_skills:
                            try:
                                success = self.db_manager.disconnect_profession_from_skills(
                                    selected_profession['title'],
                                    [skill['title'] for skill in selected_connected_skills],
                                    warnings_fn=st.warning
                                )
                                if success:
                                    st.success(labels["success_disconnect_skills_from_profession"].format(
                                        profession_title=selected_profession[language_key]
                                    ))
                                    # st.rerun()
                            except Exception as e:
                                st.error(f"Error disconnecting skills: {e}")

                    if st.button('<< ' + labels["connect_skill"], key="connect_skill_to_profession_button"):
                        # Extract the skill titles from the selected items in the multiselect
                        selected_unconnected_skill_titles = [
                            formatted_skill.split(" s::")[0] for formatted_skill in selected_unconnected_skills_titles
                        ]
                        selected_unconnected_skills = [
                            skill for skill in unconnected_skills if
                            skill[language_key] in selected_unconnected_skill_titles
                        ]
                        if selected_unconnected_skills:
                            try:
                                success = self.db_manager.connect_profession_to_skills(
                                    selected_profession['title'],
                                    [skill['title'] for skill in selected_unconnected_skills],
                                    warnings_fn=st.warning
                                )
                                if success:
                                    st.success(labels["success_connect_skills_to_profession"].format(
                                        course_title=selected_profession['title']
                                    ))
                                    # st.rerun()
                            except Exception as e:
                                st.error(f"Error connecting skills: {e}")

    def manage_professions(self, language):
        labels = MENU_ITEMS[language]
        profession_tabs = st.tabs(
            [labels["add_profession"], labels["delete_profession"], labels["update_profession"],
             labels["view_related_skills"], labels["connect_to_skills"]])

        with profession_tabs[0]:
            self.add_profession(language)
        with profession_tabs[1]:
            self.delete_profession(language)
        with profession_tabs[2]:
            self.update_profession(language)
        with profession_tabs[3]:
            self.view_related_nodes(language)
        with profession_tabs[4]:
            self.connect_to_existing_skills(language)
