import streamlit as st
from constants import *


class StreamlitCourseManager:
    def __init__(self, course_manager, db_manager, embedding_manager):
        self.course_manager = course_manager
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.key_dict = {}

    def add_course(self, language):

        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['course'][language]
        process_switch = st.radio(f"{labels['select']} :", (labels['add'], labels['load_from_file']))

        if process_switch == labels['add']:
            st.subheader(labels["add_course"])
            course_title = st.text_input(properties["title"], key="add_course_title")
            course_title_fi = st.text_input(properties["title_fi"], key="add_course_title_fi")
            course_description = st.text_area(properties["description"], key="add_course_description")
            course_description_fi = st.text_area(properties["description_fi"], key="add_course_description_fi")
            course_source_code = st.text_input(properties["source_code"], key="add_course_source_code")
            course_location = st.text_input(properties["location"], key="add_course_location")
            course_skills = st.text_area(properties["skills"], key="add_course_skills")

            if st.button(label=f"€ {labels['add_course']}", key=f'add_course_button'):

                success = self.course_manager.add_course(
                    course_title=course_title,
                    course_title_fi=course_title_fi,
                    course_description=course_description,
                    course_description_fi=course_description_fi,
                    course_source_code=course_source_code,
                    course_location=course_location,
                    course_skills=course_skills,
                    warnings_fn=st.warning
                )

                if success:
                    st.success(labels["success_add_course"].format(course_title=course_title_fi))
                    st.rerun()

        if process_switch == labels['load_from_file']:
            st.subheader(labels["load_from_file"])
            uploaded_file = st.file_uploader(labels["load_from_file"], type="xlsx", key="load_courses_file_uploader")
            if uploaded_file:
                if st.button(key=f'load_course_button', label=f'€ {labels["load_course_from_file"]}'):
                    # Initialize the progress bar
                    progress_bar = st.progress(0)

                    # Define progress update function
                    def update_progress(progress):
                        progress_bar.progress(int(progress))

                    success = self.course_manager.load_courses_from_file(uploaded_file, warnings_fn=st.warning,
                                                                         progress_fn=update_progress)

                    if success:
                        st.success(labels["success_load_courses"])
                        st.rerun()

    def delete_course(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['course'][language]
        st.subheader(labels["delete_course"])

        # Fetch all courses
        all_courses = self.course_manager.get_all_courses()
        # Check if there are no courses available
        if not all_courses:
            st.warning(labels.get("no_courses_available", "No courses available to update."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        course_titles = [course[language_key] for course in all_courses]
        selected_course_display_title = st.selectbox(properties["title"], course_titles,
                                                     key="delete_course_title_selectbox")

        try:
            # Find the selected course in the list to display its properties
            selected_course = next(
                course for course in all_courses if course[language_key] == selected_course_display_title)

            # Display the properties of the selected course
            for key, display_name in properties.items():
                st.write(f"**{display_name}:** {selected_course.get(key, 'N/A')}")

            # Button to delete the course
            if st.button(labels["delete_course"], key="delete_course_button"):
                success = self.course_manager.delete_course(selected_course.get('title'), warnings_fn=st.warning)
                if success:
                    st.success(labels["success_delete_course"].format(course_title=selected_course_display_title))
                    st.rerun()
        except StopIteration:
            st.warning(f"Course '{selected_course_display_title}' not found.")

    def update_course(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['course'][language]
        st.subheader(labels["update_course"])

        # Fetch all courses
        all_courses = self.course_manager.get_all_courses()

        # Check if there are no courses available
        if not all_courses:
            st.warning(labels.get("no_courses_available", "No courses available to update."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract course titles for the select box
        course_titles = [course[language_key] for course in all_courses]

        # Select a course to update
        selected_course_display_title = st.selectbox(labels["select_course_for_update"], course_titles,
                                                     key="update_course_title_selectbox")

        try:

            # Find the selected course in the list to display its properties
            selected_course = next(
                course for course in all_courses if course[language_key] == selected_course_display_title)

            st.subheader(labels["edit_fields"])

            # Input fields for new properties, pre-filled with existing data
            new_course_title = st.text_input(f"{properties['title']} *", value=selected_course.get('title', ''),
                                             key="update_new_course_title")
            new_course_title_fi = st.text_input(f"{properties['title_fi']} *",
                                                value=selected_course.get('title_fi', ''),
                                                key="update_new_course_title_fi")
            new_course_description = st.text_area(properties['description'],
                                                  value=selected_course.get('description', ''),
                                                  key="update_new_course_description")
            new_course_description_fi = st.text_area(properties['description_fi'],
                                                     value=selected_course.get('description_fi', ''),
                                                     key="update_new_course_description_fi")
            new_course_source_code = st.text_input(properties['source_code'],
                                                   value=selected_course.get('source_code', ''),
                                                   key="update_new_course_source_code")
            new_course_location = st.text_input(properties['location'], value=selected_course.get('location', ''),
                                                key="update_new_course_location")
            new_course_skills = st.text_input(properties['skills'], value=selected_course.get('skills', ''),
                                              key="update_new_course_skills")

            # Process the update button click
            if st.button(labels["update_course"], key="update_course_button"):
                # Validate required fields
                if not new_course_title or not new_course_title_fi:
                    st.warning("Both course title and Finnish course title are required.")
                    return

                updated_properties = {
                    'title': new_course_title,
                    'title_fi': new_course_title_fi,
                    'description': new_course_description,
                    'description_fi': new_course_description_fi,
                    'location': new_course_location,
                    'source_code': new_course_source_code,
                    'skills': new_course_skills
                }

                # Perform the course update
                success = self.course_manager.update_course(selected_course['title'], warnings_fn=st.warning,
                                                            **updated_properties)
                if success:
                    st.success(labels["success_update_course"].format(course_title=selected_course['title']))
                    st.rerun()
        except StopIteration:
            st.warning(f"Course '{selected_course_display_title}' not found.")

    def view_related_nodes(self, language, warnings_fn=None):
        labels = MENU_ITEMS[language]
        skill_properties = NODE_PROPERTIES['skill'][language]

        st.subheader(labels["view_related_nodes"])

        session_state_keys = ['selected_course_display_title', 'selected_course_title_en', 'selected_node_title']

        for key in session_state_keys:
            if key not in st.session_state:
                st.session_state[key] = None

        # Fetch all courses
        all_courses = self.course_manager.get_all_courses()

        if not all_courses:
            st.warning(labels.get("no_courses_available", "No courses available to view related nodes."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract course titles for the select box
        courses_titles = [course[language_key] for course in all_courses]

        # Select a course to view related nodes
        selected_course_title = st.selectbox(labels["select_course"], courses_titles, index=None,
                                             key="view_related_nodes_course_selectbox")

        if not selected_course_title:
            st.warning(labels.get("select_course_warning", "Please select a course to view related nodes."))
            return

        if selected_course_title and st.button(labels["view"], key="view_related_to_course_nodes_button"):
            st.session_state.selected_course_display_title = selected_course_title
            st.session_state.selected_course_title_en = next(
                (course['title'] for course in all_courses if course[language_key] == selected_course_title), None)

        if st.session_state.selected_course_display_title and st.session_state.selected_course_title_en:

            related_nodes = self.db_manager.get_related_nodes(DEFAULT_COURSE_LABEL,
                                                              st.session_state.selected_course_title_en)

            skills = {node_id: node_info for node_id, node_info in related_nodes.items() if
                      DEFAULT_SKILL_LABEL in node_info['labels']}
            new_skills = {node_id: node_info for node_id, node_info in related_nodes.items() if
                          DEFAULT_NEW_SKILL_LABEL in node_info['labels']}

            if skills:
                st.subheader(labels["related_skills"])
                for node_id, node_info in skills.items():
                    for key, display_name in skill_properties.items():
                        st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")
                    st.write(f"**{labels['relationship_type']}:** {node_info.get('relationship_type', 'N/A')}")

                    if st.button(
                            f"{labels['remove_connection_between']} {node_info.get(language_key, 'N/A')} {labels['and']} {st.session_state.selected_course_display_title}",
                            key=f"disconnect_skill_{node_id}"):
                        st.session_state.selected_node_title = node_info['title']

            if new_skills:
                st.subheader(labels["related_new_skills"])
                for node_id, node_info in new_skills.items():
                    for key, display_name in skill_properties.items():
                        st.write(f"**{display_name}:** {node_info.get(key, 'N/A')}")
                    st.write(f"**{labels['relationship_type']}:** {node_info.get('relationship_type', 'N/A')}")

                    if st.button(
                            f"{labels['remove_connection_between']} {node_info.get(language_key, 'N/A')} {labels['and']} {st.session_state.selected_course_display_title}",
                            key=f"disconnect_skill_{node_id}"):
                        st.session_state.selected_node_title = node_info['title']

        # Perform the disconnection after a button is pressed
        if st.session_state.selected_node_title and st.session_state.selected_course_title_en:
            st.write("Deleting connection with :", st.session_state.selected_node_title)  # Streamlit feedback

            self.db_manager.disconnect_course_from_skills(st.session_state.selected_course_title_en,
                                                          [st.session_state.selected_node_title], warnings_fn)
            st.success(
                f"Disconnected {st.session_state.selected_course_display_title} from {st.session_state.selected_node_title}")

            st.session_state.selected_node_title = None
            st.rerun()

    def connect_to_existing_skills(self, language):
        labels = MENU_ITEMS[language]
        course_properties = NODE_PROPERTIES['course'][language]

        # Fetch all courses
        all_courses = self.course_manager.get_all_courses()

        # Check if there are no courses available
        if not all_courses:
            st.warning(labels.get("no_courses_available", "No courses available to update."))
            return

        # Determine the correct language key
        language_key = 'title_fi' if language == 'Finnish' else 'title'

        # Extract course titles for the select box
        course_titles = [course[language_key] for course in all_courses]
        # Select a course to update
        selected_course_display_title = st.selectbox(labels["select_course_for_update"], course_titles,
                                                     key="connect_to_skills_course_title_selectbox")

        # Find the selected course in the list to display its properties
        if selected_course_display_title:
            selected_course = next(
                (course for course in all_courses if course[language_key] == selected_course_display_title), None)

            for key, display_name in course_properties.items():
                st.write(f"**{display_name}:** {selected_course.get(key, 'N/A')}")

            if selected_course:
                try:
                    connected_skills = self.db_manager.get_skills_connected_to_course(selected_course['title'])
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
                        selected_course['title'],
                        selected_course['description'],
                        NUM_SIMILAR_SKILLS_FOR_COURSE_CONNECTION
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

                course_col1, course_col2, course_col3 = st.columns([1, 0.8, 1])

                with course_col1:
                    st.subheader(labels["connected_skills"])
                    # selected_connected_skills_titles = [
                    #     title for title in connected_skills_titles if st.checkbox(title, key=f"connected_{title}")
                    # ]
                    selected_connected_skills_titles = [title for idx, title in enumerate(connected_skills_titles)
                                                        if st.checkbox(title, key=f"connected_{idx}_{title}")]
                with course_col3:
                    st.subheader(labels["unconnected_skills"])
                    selected_unconnected_skills_titles = st.multiselect(" ", formatted_unconnected_skills_titles,
                                                                        key="unconnected_skills_multiselect")

                st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add space before the buttons

                with course_col2:
                    if st.button(labels["disconnect_skill"] + ' >>', key="disconnect_skill_from_course_button_tab5"):
                        selected_connected_skills = [
                            skill for skill in connected_skills if
                            skill[language_key] in selected_connected_skills_titles
                        ]
                        if selected_connected_skills:
                            try:

                                success = self.db_manager.disconnect_course_from_skills(
                                    selected_course['title'],
                                    [skill['title'] for skill in selected_connected_skills],
                                    warnings_fn=st.warning
                                )
                                if success:
                                    st.success(labels["success_disconnect_skills_from_course"].format(
                                        course_title=selected_course[language_key]
                                    ))
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error disconnecting skills: {e}")

                    if st.button('<< ' + labels["connect_skill"], key="connect_skill_to_course_button"):
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
                                success = self.db_manager.connect_course_to_skills(
                                    selected_course['title'],
                                    [skill['title'] for skill in selected_unconnected_skills],
                                    warnings_fn=st.warning
                                )
                                if success:
                                    st.success(labels["success_connect_skills_to_course"].format(
                                        course_title=selected_course['title']
                                    ))
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error connecting skills: {e}")

    def manage_courses(self, language):
        labels = MENU_ITEMS[language]
        course_tabs = st.tabs(
            [labels["add_course"], labels["delete_course"], labels["update_course"], labels["view_related_skills"],
             labels["connect_to_existing_skills"]])

        with course_tabs[0]:
            self.add_course(language)
        with course_tabs[1]:
            self.delete_course(language)
        with course_tabs[2]:
            self.update_course(language)
        with course_tabs[3]:
            self.view_related_nodes(language)
        with course_tabs[4]:
            self.connect_to_existing_skills(language)
