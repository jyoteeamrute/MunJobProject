import re
import streamlit as st
from constants import *


class StreamlitSkillManager:
    def __init__(self, skill_manager, db_manager, embedding_manager):
        self.skill_manager = skill_manager
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.key_dict = {}

    def add_skill(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['skill'][language]
        process_switch = st.radio(f"{labels['select']} :", (labels['add'], labels['load_from_file']))

        if process_switch == labels['add']:
            st.subheader(labels["add_skill"])
            skill_source_id = st.text_input(properties["source_id"], key="add_skill_source_id")
            skill_source_code = st.text_input(properties["source_code"], key="add_skill_source_code")
            skill_title = st.text_input(f"{properties['title']} *", key="add_skill_title")
            skill_title_fi = st.text_input(f"{properties['title']} (FI) *", key="add_skill_title_fi")
            skill_description = st.text_area(properties["description"], key="add_skill_description")
            skill_description_fi = st.text_area(f"{properties['description']} (FI)", key="add_skill_description_fi")
            skill_type = st.selectbox(properties["type"], [DEFAULT_SKILL_TYPE_PROFESSIONAL,
                                                           DEFAULT_SKILL_TYPE_IT,
                                                           DEFAULT_SKILL_TYPE_LANGUAGE,
                                                           DEFAULT_SKILL_TYPE_SOFT],
                                      key="add_skill_type")

            if st.button(labels["add_skill"], key="add_skill_button"):

                if not skill_title or not skill_title_fi:
                    st.warning(labels["title_empty_warning"])
                else:
                    success = self.skill_manager.add_skill(skill_source_id, skill_source_code, skill_title,
                                                           skill_title_fi, skill_description, skill_description_fi,
                                                           skill_label=DEFAULT_SKILL_LABEL, skill_type=skill_type,
                                                           warnings_fn=st.warning)
                    if success:
                        st.success(labels["success_add_skill"].format(skill_title=skill_title))
                        st.rerun()

        elif process_switch == labels['load_from_file']:
            st.subheader(labels["load_from_file"])
            uploaded_file = st.file_uploader(labels["load_from_file"], type="xlsx", key="load_skills_file_uploader")
            if uploaded_file:
                if st.button(labels["load_from_file"], key="load_skills_button"):
                    # Initialize the progress bar
                    progress_bar = st.progress(0)

                    # Define progress update function
                    def update_progress(progress):
                        progress_bar.progress(int(progress))

                    success = self.skill_manager.load_skills_from_file(uploaded_file, warnings_fn=st.warning,
                                                                       progress_fn=update_progress)
                    if success:
                        st.success(labels["success_load_skills"])
                        st.rerun()

    def delete_skill(self, skill_title, sfn=None):
        try:
            # Assume delete operation here
            success = True  # Change based on actual logic
            if success:
                if sfn:
                    sfn(f"Successfully deleted skill '{skill_title}'.")
            else:
                if sfn:
                    sfn(f"Failed to delete skill '{skill_title}'.")
            return success
        except Exception as e:
            if sfn:
                sfn(f"Error: {str(e)}")
            return False

    def delete_skill(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['skill'][language]
        st.subheader(labels["delete_skill"])

        # Fetch all skills
        all_skills = self.skill_manager.get_all_skills()

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        # Extract skill titles for the select box
        skill_titles = [skill[language_key] for skill in all_skills]
        # Clean course titles by removing special characters and quotation marks
        cleaned_skill_titles = [re.sub(r'[\'\"“”]', '', title.strip().lower()) for title in skill_titles]

        # Pair the cleaned titles with the original titles
        skill_pairs = zip(cleaned_skill_titles, skill_titles)

        # Sort based on the cleaned titles
        sorted_skill_pairs = sorted(skill_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_skill_titles = [pair[1] for pair in sorted_skill_pairs]
        # Select a skill to delete
        selected_skill_title = st.selectbox(properties["title"], sorted_skill_titles, key="delete_skill_title_selectbox")
        try:
            # Find the selected skill in the list to display its properties
            selected_skill = next(skill for skill in all_skills if skill[language_key] == selected_skill_title)

            properties = NODE_PROPERTIES.get('skill', {}).get(language, {})
            for key, display_name in properties.items():
                st.write(f"**{display_name}:** {selected_skill.get(key, 'N/A')}")

            # Button to delete the skill
            if st.button(labels["delete_skill"], key="delete_skill_button"):
                success = self.skill_manager.delete_skill(selected_skill.get('title'))
                if success:
                    st.success(labels["success_delete_skill"].format(skill_title=selected_skill_title))
                    st.rerun()
        except StopIteration:
            st.warning(f"Skill '{selected_skill_title}' not found.")

    def update_skill(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['skill'][language]
        st.subheader(labels["update_skill"])

        # Fetch all skills
        all_skills = self.skill_manager.get_all_skills()

        if not all_skills:
            st.warning(labels.get("no_skills_available", "No skills available to update."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract skill titles for the select box
        skill_titles = [skill[language_key] for skill in all_skills]
        # Clean course titles by removing special characters and quotation marks
        cleaned_skill_titles = [re.sub(r'[\'\"“”]', '', title.strip().lower()) for title in skill_titles]

        # Pair the cleaned titles with the original titles
        skill_pairs = zip(cleaned_skill_titles, skill_titles)

        # Sort based on the cleaned titles
        sorted_skill_pairs = sorted(skill_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_skill_titles = [pair[1] for pair in sorted_skill_pairs]
        # Select a skill to update
        selected_skill_title = st.selectbox(labels["select_skill_for_update"], sorted_skill_titles,
                                            key="update_skill_title_selectbox")

        st.subheader(labels["edit_fields"])

        # Find the selected skill in the list to display its properties
        selected_skill = next(skill for skill in all_skills if skill[language_key] == selected_skill_title)

        # Input fields for new properties, pre-filled with existing data
        new_skill_title = st.text_input(f"{properties['title']} *",
                                        value=selected_skill.get('title', ''),
                                        key="update_new_skill_title")
        new_skill_description = st.text_area(properties['description'],
                                             value=selected_skill.get('description', ''),
                                             key="update_new_skill_description")
        new_skill_type = st.selectbox(properties['type'], [DEFAULT_SKILL_TYPE_PROFESSIONAL,
                                                           DEFAULT_SKILL_TYPE_IT,
                                                           DEFAULT_SKILL_TYPE_LANGUAGE,
                                                           DEFAULT_SKILL_TYPE_SOFT],
                                      index=[DEFAULT_SKILL_TYPE_PROFESSIONAL,
                                             DEFAULT_SKILL_TYPE_IT,
                                             DEFAULT_SKILL_TYPE_LANGUAGE,
                                             DEFAULT_SKILL_TYPE_SOFT].index(
                                          selected_skill.get('type', DEFAULT_SKILL_TYPE_PROFESSIONAL)),
                                      key="update_new_skill_type")
        new_skill_source_id = st.text_input(properties['source_id'], value=selected_skill.get('source_id', ''),
                                            key="update_new_skill_source_id")
        new_skill_source_code = st.text_input(properties['source_code'],
                                              value=selected_skill.get('source_code', ''),
                                              key="update_new_skill_source_code")
        new_skill_title_fi = st.text_input(f"{properties['title']} (FI) *", value=selected_skill.get('title_fi', ''),
                                           key="update_new_skill_title_fi")
        new_skill_description_fi = st.text_area(f"{properties['description']} (FI)",
                                                value=selected_skill.get('description_fi', ''),
                                                key="update_new_skill_description_fi")

        if st.button(labels["update_skill"], key="update_skill_button"):
            # Validate required fields
            if not new_skill_title or not new_skill_title_fi:
                st.warning("Both skill title and Finnish skill title are required.")
                return

            updated_properties = {
                'title': new_skill_title,
                'description': new_skill_description,
                'type': new_skill_type,
                'source_id': new_skill_source_id,
                'source_code': new_skill_source_code,
                'title_fi': new_skill_title_fi,
                'description_fi': new_skill_description_fi
            }
            if selected_skill_title == 'none':#handle if title is none for finish
                selected_skill_title = new_skill_title
            success = self.skill_manager.update_skill(selected_skill_title, warnings_fn=st.warning,
                                                      **updated_properties)
            if success:
                st.success(labels["success_update_skill"].format(skill_title=selected_skill_title))
                st.rerun()

    def view_related_nodes(self, language, warnings_fn=None):
        labels = MENU_ITEMS[language]
        person_properties = NODE_PROPERTIES['person'][language]
        profession_properties = NODE_PROPERTIES['profession'][language]
        course_properties = NODE_PROPERTIES['course'][language]

        st.subheader(labels["view_related_nodes"])

        session_state_keys = ['selected_skill_display_title', 'selected_skill_title_en', 'selected_node_title_or_name',
                              'selected_node_label']
        for key in session_state_keys:
            if key not in st.session_state:
                st.session_state[key] = None

        # Fetch all skills
        all_skills = self.skill_manager.get_all_skills()

        if not all_skills:
            st.warning(labels.get("no_skills_available", "No skills available to view related nodes."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract skill titles for the select box
        skill_titles = [skill[language_key] for skill in all_skills]
        # Clean course titles by removing special characters and quotation marks
        cleaned_skill_titles = [re.sub(r'[\'\"“”]', '', title.strip().lower()) for title in skill_titles]

        # Pair the cleaned titles with the original titles
        skill_pairs = zip(cleaned_skill_titles, skill_titles)

        # Sort based on the cleaned titles
        sorted_skill_pairs = sorted(skill_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_skill_titles = [pair[1] for pair in sorted_skill_pairs]
        # Select a skill to view related nodes
        selected_skill_title = st.selectbox(labels["select_skill"], sorted_skill_titles, index=None,
                                            key="view_related_nodes_skill_selectbox")

        if not selected_skill_title:
            st.warning(labels.get("select_skill_warning", "Please select a skill to view related nodes."))
            return

        if selected_skill_title and st.button(labels["view"], key="view_related_nodes_button"):
            st.session_state.selected_skill_display_title = selected_skill_title
            st.session_state.selected_skill_title_en = next(
                (skill['title'] for skill in all_skills if skill[language_key] == selected_skill_title), None)

        if st.session_state.selected_skill_display_title and st.session_state.selected_skill_title_en:
            # Fetch related nodes for both labels
            related_nodes_skill = self.db_manager.get_related_nodes(DEFAULT_SKILL_LABEL,
                                                                    st.session_state.selected_skill_title_en)
            related_nodes_new_skill = self.db_manager.get_related_nodes(DEFAULT_NEW_SKILL_LABEL,
                                                                        st.session_state.selected_skill_title_en)

            if len(related_nodes_skill) == 0 and len(related_nodes_new_skill) == 0:
                st.write(labels['no_related_nodes'])

            # Merge the dictionaries
            related_nodes = {**related_nodes_skill, **related_nodes_new_skill}

            courses = {node_id: node_info for node_id, node_info in related_nodes.items() if
                       DEFAULT_COURSE_LABEL in node_info['labels']}
            professions = {node_id: node_info for node_id, node_info in related_nodes.items() if
                           DEFAULT_PROFESSION_LABEL in node_info['labels']}
            persons = {node_id: node_info for node_id, node_info in related_nodes.items() if
                       DEFAULT_PERSON_LABEL in node_info['labels']}

            # Display related courses
            if courses:
                st.subheader(labels["related_courses"])
                for node_id, node_info in courses.items():
                    for key, display_name in course_properties.items():
                        st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")

                    if st.button(
                            f"{labels['remove_connection_between']} {node_info.get(language_key, 'N/A')} {labels['and']} {st.session_state.selected_skill_display_title}",
                            key=f"disconnect_course_{node_id}"):
                        st.session_state.selected_node_title_or_name = node_info['title']
                        st.session_state.selected_node_label = 'course'

            # Display related professions
            if professions:
                st.subheader(labels["related_professions"])
                for node_id, node_info in professions.items():
                    for key, display_name in profession_properties.items():
                        st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")

                    if st.button(
                            f"{labels['remove_connection_between']} {node_info.get(language_key, 'N/A')} {labels['and']} {st.session_state.selected_skill_display_title}",
                            key=f"disconnect_profession_{node_id}"):
                        st.session_state.selected_node_title_or_name = node_info['title']
                        st.session_state.selected_node_label = 'profession'

            # Display related persons
            if persons:
                st.subheader(labels["related_persons"])
                for node_id, node_info in persons.items():
                    for key, display_name in person_properties.items():
                        st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")

                    if st.button(
                            f"{labels['remove_connection_between']} {node_info.get('name', 'N/A')} {labels['and']} {st.session_state.selected_skill_display_title}",
                            key=f"disconnect_person_{node_id}"):
                        st.session_state.selected_node_title_or_name = node_info['name']
                        st.session_state.selected_node_label = 'person'

        # Perform the disconnection after a button is pressed
        if st.session_state.selected_node_title_or_name and st.session_state.selected_skill_title_en and st.session_state.selected_node_label:
            st.write("Deleting connection with :", st.session_state.selected_node_title_or_name)  # Streamlit feedback

            if st.session_state.selected_node_label == 'course':
                self.db_manager.disconnect_course_from_skills(st.session_state.selected_node_title_or_name,
                                                              [st.session_state.selected_skill_title_en], warnings_fn)
                st.success(
                    f"Disconnected {st.session_state.selected_skill_display_title} from {st.session_state.selected_node_title_or_name}")

                st.session_state.selected_node_title_or_name = None
                st.session_state.selected_node_label = None
                st.rerun()

            if st.session_state.selected_node_label == 'profession':
                self.db_manager.disconnect_profession_from_skills(st.session_state.selected_node_title_or_name,
                                                                  [st.session_state.selected_skill_title_en],
                                                                  warnings_fn)
                st.success(
                    f"Disconnected {st.session_state.selected_skill_display_title} from {st.session_state.selected_node_title_or_name}")

                st.session_state.selected_node_title_or_name = None
                st.session_state.selected_node_label = None
                st.rerun()

            if st.session_state.selected_node_label == 'person':
                self.db_manager.disconnect_person_from_skills(st.session_state.selected_node_title_or_name,
                                                              [st.session_state.selected_skill_title_en], warnings_fn)
                st.success(
                    f"Disconnected {st.session_state.selected_skill_display_title} from {st.session_state.selected_node_title_or_name}")

                st.session_state.selected_node_title_or_name = None
                st.session_state.selected_node_label = None

                st.rerun()

    def view_new_skills(self, language):

        def get_gpt_description(title):
            # Fetch related nodes
            courses_description = self.db_manager.get_description_of_related_courses(title, DEFAULT_NEW_SKILL_LABEL)
            professions_description = self.db_manager.get_description_of_related_professions(title,
                                                                                             DEFAULT_NEW_SKILL_LABEL)
            return f"{courses_description} {professions_description}"

        labels = MENU_ITEMS.get(language, {})
        properties = NODE_PROPERTIES.get('skill', {}).get(language, {})

        # Fetch all skills
        all_new_skills = self.skill_manager.get_all_new_skills()
        all_skills = self.skill_manager.get_all_skills()

        if not all_new_skills:
            st.warning(labels.get("no_skills_available", "No new skills available."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract skill titles for the select box
        skill_titles = [skill.get(language_key, "") for skill in all_new_skills]
        # Clean course titles by removing special characters and quotation marks
        cleaned_skill_titles = [re.sub(r'[\'\"“”]', '', title.strip().lower()) for title in skill_titles]

        # Pair the cleaned titles with the original titles
        skill_pairs = zip(cleaned_skill_titles, skill_titles)

        # Sort based on the cleaned titles
        sorted_skill_pairs = sorted(skill_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_skill_titles = [pair[1] for pair in sorted_skill_pairs]
        # Select a skill to view related nodes
        selected_skill_title = st.selectbox(labels.get('view_related_nodes', 'View Related Nodes'), sorted_skill_titles,
                                            key="get_related_to_new_skill_nodes_skill_selectbox")

        selected_skill = next(
            (skill for skill in all_new_skills if skill.get(language_key, "") == selected_skill_title), None)
        if selected_skill:
            st.subheader(f"{labels.get('new_skills', 'New Skills')}: {selected_skill.get(language_key, '')}")

            # Keys for session state
            key_en = f"description_en_{selected_skill.get('title', '')}"
            key_fi = f"description_fi_{selected_skill.get('title', '')}"

            # Initialize session state if not present
            if key_en not in st.session_state:
                st.session_state[key_en] = selected_skill.get('description', '')
            if key_fi not in st.session_state:
                st.session_state[key_fi] = selected_skill.get('description_fi', '')

            # Text area for descriptions
            st.text_area(properties.get('description', 'Description'), value=st.session_state[key_en])
            st.text_area(properties.get('description_fi', 'Description (FI)'),
                         value=st.session_state[key_fi])

            col11, col12 = st.columns(2)
            if f'button_desc_en_{selected_skill["title"]}_pressed' not in st.session_state:
                st.session_state[f'button_desc_en_{selected_skill["title"]}_pressed'] = False

            if f'button_desc_fi_{selected_skill["title"]}_pressed' not in st.session_state:
                st.session_state[f'button_desc_fi_{selected_skill["title"]}_pressed'] = False

            with col11:
                if st.button(key=f'button_desc_en_{selected_skill["title"]}',
                            label=f"€ {labels.get('generate_gpt_desc_en', 'Generate GPT Description (EN)')}"):
                    st.session_state[f'button_desc_en_{selected_skill["title"]}_pressed'] = True

            with col12:
                if st.button(key=f'button_desc_fi_{selected_skill["title"]}',
                             label = f"€ {labels.get('generate_gpt_desc_fi', 'Generate GPT Description (FI)')}"):
                    st.session_state[f'button_desc_fi_{selected_skill["title"]}_pressed'] = True

            if st.session_state[f'button_desc_en_{selected_skill["title"]}_pressed']:
                description_en_value = self.skill_manager.gpt_client.generate_skill_description_english(
                    selected_skill['title'], get_gpt_description(selected_skill['title']))
                st.session_state[key_en] = description_en_value

                st.session_state[f'button_desc_en_{selected_skill["title"]}_pressed'] = False
                st.rerun()

            if st.session_state[f'button_desc_fi_{selected_skill["title"]}_pressed']:
                description_fi_value = self.skill_manager.gpt_client.generate_skill_description_finnish(
                    selected_skill['title'], get_gpt_description(selected_skill['title']))
                st.session_state[key_fi] = description_fi_value
                st.session_state[f'button_desc_fi_{selected_skill["title"]}_pressed'] = False
                st.rerun()

            # Construct the update dictionary with all fields from selected_skill, updating description and description_fi
            updated_skill = {**selected_skill, 'description': st.session_state[key_en],
                             'description_fi': st.session_state[key_fi]}

            # Update the skill using the updated_skill dictionary
            self.skill_manager.update_skill(selected_skill['title'], **updated_skill)

            related_nodes = self.db_manager.get_related_nodes(DEFAULT_NEW_SKILL_LABEL, selected_skill['title'])

            if related_nodes:
                st.subheader(labels.get("related_programs_or_professions", "Related Programs or Professions"))
                for node_id, node_info in related_nodes.items():
                    node_labels = node_info.get('labels', [])
                    if DEFAULT_COURSE_LABEL in node_labels:
                        course_properties = NODE_PROPERTIES.get('course', {}).get(language, {})
                        for key, display_name in course_properties.items():
                            st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")
                    elif DEFAULT_PROFESSION_LABEL in node_labels:
                        profession_properties = NODE_PROPERTIES.get('profession', {}).get(language, {})
                        for key, display_name in profession_properties.items():
                            st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")
                    # Display relationship type
                    st.write(f"**Relationship Type:** {node_info.get('relationship_type', 'N/A')}")

            if not len(related_nodes):
                st.write(labels['no_related_nodes'])

            if st.button(labels.get('accept_new_skill', 'Accept New Skill'), key=f"confirm_{selected_skill['title']}"):

                success = self.skill_manager.confirm_skill(selected_skill['title'], warnings_fn=st.warning)
                if success:
                    st.success(f"Skill '{selected_skill['title']}' confirmed.")
                    st.rerun()

            if st.button(labels.get('delete_skill', 'Delete Skill'), key=f"reject_{selected_skill['title']}"):
                success = self.skill_manager.delete_skill(selected_skill['title'], warnings_fn=st.warning)
                if success:
                    st.success(f"Skill '{selected_skill['title']}' deleted.")
                    st.rerun()

            # Show similar skills
            st.subheader(f"{labels.get('skills_similar_to', 'Skills similar to')} {selected_skill[language_key]}")
            similar_skills = self.embedding_manager.find_top_similar_skills(selected_skill['title'],
                                                                            selected_skill.get('description', ''),
                                                                            top_similar=DISPLAY_SIMILAR_SKILLS_LIMIT)
            similar_skills = similar_skills[1:]

            description_key = 'description' if language_key == 'title' else 'description_fi'

            if similar_skills:
                for similar_skill, similarity in similar_skills:
                    similar_skill_data = next((skill for skill in all_skills if skill['title'] == similar_skill), None)
                    if similar_skill_data:
                        similar_skill_title = similar_skill_data.get(language_key, 'N/A')
                        similar_skill_description = similar_skill_data.get(description_key, 'N/A')
                        st.markdown(f"<h6><b>{similar_skill_title}</b></h6>", unsafe_allow_html=True)
                        st.write(f"Similarity: {similarity:.2f}")
                        st.write(f"Description: {similar_skill_description}")

    def manage_skill(self, language):
        labels = MENU_ITEMS[language]
        skill_tabs = st.tabs(
            [labels["add_skill"], labels["delete_skill"], labels["update_skill"], labels["view_related_nodes"],
             labels["view_new_skills"]])

        with skill_tabs[0]:
            self.add_skill(language)
        with skill_tabs[1]:
            self.delete_skill(language)
        with skill_tabs[2]:
            self.update_skill(language)
        with skill_tabs[3]:
            self.view_related_nodes(language)
        with skill_tabs[4]:
            self.view_new_skills(language)
