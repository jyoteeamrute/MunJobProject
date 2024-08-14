import re
import streamlit as st
from constants import *
import matplotlib.pyplot as plt
import networkx as nx


class StreamlitPersonManager:
    def __init__(self, person_manager, db_manager):
        self.person_manager = person_manager
        self.db_manager = db_manager

    def add_person(self, language, warnings_fn=None):

        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['person'][language]

        # st.subheader(f"{labels["add_person"]} *")
        st.subheader(f"{labels['add_person']} *")

        person_name = st.text_input(properties["name"], key="add_person_name")

        all_skills = self.db_manager.get_all_skills()

        if not all_skills:
            if warnings_fn:
                st.warning(labels.get("no_skills_available", "No skills available."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        all_skill_titles = [skill[language_key] for skill in all_skills]
        #sorted a all_skill_titles
        cleaned_all_skill_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in all_skill_titles]

        # Pair he cleaned title with the original title
        all_skil_pairs = zip(cleaned_all_skill_titles, all_skill_titles)

        # Sort based on the cleaned title
        sorted_all_skil_pairs= sorted(all_skil_pairs, key=lambda x: x[0])
        # Extract the sorted original title
        sorted_sorted_all_skil_title = [pair[1] for pair in sorted_all_skil_pairs]
        selected_skills_titles = st.multiselect(labels["select_skills_for_person"], sorted_sorted_all_skil_title,
                                                key="selected_skill_title")
        selected_connected_skills = [
            skill for skill in all_skills if
            skill[language_key] in selected_skills_titles
        ]

        if st.button(labels['add_person_with_selected_skills_to_the_graph'], key=f'add_person_button'):
            if not person_name:
                if warnings_fn:
                    st.warning(labels["person_name_empty_warning"])
                return

            success = self.person_manager.add_person(
                person_name,
                selected_connected_skills,
                warnings_fn=st.warning
            )
            if success:
                st.success(labels["success_add_person"].format(person_name=person_name))
                st.rerun()

    def delete_person(self, language):
        labels = MENU_ITEMS[language]
        properties = NODE_PROPERTIES['skill'][language]
        st.subheader(labels["delete_person"])

        persons_names = sorted(self.person_manager.get_all_persons()) 
        if not persons_names:
            st.warning(labels.get("no_persons_available", "No persons available."))
            return
        #sorted a person names list
        cleaned_persons_names = [re.sub(r'\s+', ' ', title.strip().lower()) for title in persons_names]

        # Pair he cleaned names with the original name
        person_pairs = zip(cleaned_persons_names, persons_names)

        # Sort based on the cleaned name
        sorted_person_pairs = sorted(person_pairs, key=lambda x: x[0])

        # Extract the sorted original name
        sorted_person_name = [pair[1] for pair in sorted_person_pairs]
        selected_person_name = st.selectbox(labels["name"], sorted_person_name, index=None,
                                            key="delete_person_name_selectbox")

        try:
            if selected_person_name:
                person_skills = self.db_manager.get_person_skills(selected_person_name)
                st.subheader(labels["persons_skills"])

                for skill in person_skills:

                    for key, display_name in properties.items():
                        st.markdown(f"<b>{display_name}:</b> {skill.get(key, 'N/A')}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                if st.button(labels["delete_person"], key="delete_person_button"):
                    success = self.person_manager.delete_person(selected_person_name, warnings_fn=st.warning)
                    if success:
                        st.success(labels["success_delete_person"].format(person_name=selected_person_name))
                        st.rerun()
        except StopIteration:
            st.warning(f"Person '{selected_person_name}' not found.")

    def update_person(self, language):
        labels = MENU_ITEMS[language]
        persons_names = self.person_manager.get_all_persons()

        if not persons_names:
            st.warning(labels.get("no_persons_available", "No persons available."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'
           
        #sorted a person names list
        cleaned_persons_names = [re.sub(r'\s+', ' ', title.strip().lower()) for title in persons_names]

        # Pair he cleaned names with the original name
        person_pairs = zip(cleaned_persons_names, persons_names)

        # Sort based on the cleaned name
        sorted_person_pairs = sorted(person_pairs, key=lambda x: x[0])

        # Extract the sorted original name
        sorted_person_name = [pair[1] for pair in sorted_person_pairs]
        selected_person_name = st.selectbox(labels["select_person"], sorted_person_name, index=None,
                                            key="update_person_name_selectbox")
        if not selected_person_name:
            st.warning(labels.get("person_name_empty_warning", "Please select a person."))
        else:

            new_person_name = st.text_input(labels["edit_name"], selected_person_name, key="person_name_update")
            if st.button(labels['update_person'], key="update_person_button"):
                if not new_person_name:
                    st.warning(labels["person_name_empty_warning"])
                else:
                    success = self.person_manager.update_person(selected_person_name, new_person_name,
                                                                warnings_fn=st.warning)
                    if success:
                        st.success(labels["success_update_person"].format(person_name=selected_person_name))
                        st.rerun()

            person_skills = self.db_manager.get_person_skills(selected_person_name)
            all_skills = self.db_manager.get_all_skills()

            # Filter out skills with empty title or title_fi
            person_skills = [skill for skill in person_skills if skill.get('title') and skill.get('title_fi')]

            unconnected_skills = [skill for skill in all_skills if
                                  skill not in person_skills and skill.get('title') and skill.get('title_fi')]

            unconnected_skills_titles = [skill[language_key] for skill in unconnected_skills]

            person_col1, person_col2, person_col3 = st.columns([1, 0.8, 1])

            with person_col1:
                st.subheader(labels["person_has_skills"])
                selected_connected_skills_titles = [
                    skill[language_key] for skill in person_skills if
                    st.checkbox(skill[language_key], key=f"person_has_{skill['source_id']}")
                ]

            with person_col3:
                st.subheader(labels["person_has_no_skills"])
                selected_unconnected_skills_titles = st.multiselect(" ", unconnected_skills_titles,
                                                                    key="unconnected_to_person_skills_multiselect")

            with person_col2:
                if st.button(labels["disconnect_skill"] + ' >>', key="disconnect_skill_from_person_button"):
                    selected_connected_skills = [
                        skill for skill in person_skills if
                        skill[language_key] in selected_connected_skills_titles
                    ]
                    if selected_connected_skills:
                        try:
                            success = self.db_manager.disconnect_person_from_skills(
                                selected_person_name,
                                [skill['title'] for skill in selected_connected_skills],
                                warnings_fn=st.warning
                            )
                            if success:
                                st.success(labels["success_disconnect_person_from_course"])
                                # st.rerun()
                        except Exception as e:
                            st.error(f"Error disconnecting skills: {e}")

                if st.button('<< ' + labels["connect_skill"], key="connect_skill_to_person_button"):
                    selected_unconnected_skills = [
                        skill for skill in unconnected_skills if
                        skill[language_key] in selected_unconnected_skills_titles
                    ]
                    if selected_unconnected_skills:
                        try:
                            success = self.db_manager.connect_person_to_skills(
                                selected_person_name,
                                [skill['title'] for skill in selected_unconnected_skills],
                                warnings_fn=st.warning
                            )
                            if success:
                                st.success(labels["success_connect_skills_to_person"])
                                # st.rerun()
                        except Exception as e:
                            st.error(f"Error connecting skills: {e}")

    def suggest_training_program(self, language):
        labels = MENU_ITEMS[language]
        course_properties = NODE_PROPERTIES['course'][language]
        skill_properties = NODE_PROPERTIES['skill'][language]

        st.subheader(labels["suggest_training_program"])

        persons_names = self.person_manager.get_all_persons()
     
        if not persons_names:
            st.warning(labels.get("no_persons_available", "No persons available."))
            return

        all_professions = self.db_manager.get_all_professions()

        if not all_professions:
            st.warning(labels.get("no_professions_available", "No professions available."))
            return

        if language == 'Finnish':
            language_key = 'title_fi'
        else:
            language_key = 'title'

        profession_titles = [profession[language_key] for profession in all_professions]
        cleaned_profession_titles = [re.sub(r'\s+', ' ', title.strip().lower()) for title in profession_titles]

        # Pair he cleaned titles with the original titles
        profession_pairs = zip(cleaned_profession_titles, profession_titles)

        # Sort based on the cleaned titles
        sorted_profession_pairs = sorted(profession_pairs, key=lambda x: x[0])

        # Extract the sorted original titles
        sorted_profession_titles = [pair[1] for pair in sorted_profession_pairs]
     
        #!!!sorted a person names list
        cleaned_persons_names = [re.sub(r'\s+', ' ', title.strip().lower()) for title in persons_names]

        # Pair he cleaned names with the original name
        person_pairs = zip(cleaned_persons_names, persons_names)

        # Sort based on the cleaned name
        sorted_person_pairs = sorted(person_pairs, key=lambda x: x[0])

        # Extract the sorted original name
        sorted_person_name = [pair[1] for pair in sorted_person_pairs]
        selected_person_name = st.selectbox(labels["select_client_for_suggestion"], sorted_person_name, index=None,
                                            key="person_name_selecct_skill_gap")
        selected_profession_title = st.selectbox(labels["select_profession_for_suggestion"], sorted_profession_titles,
                                                 index=None, key="profession_select_skill_gap")

        if st.button(labels["suggest_training_program"], key="suggest_training_program_button"):
            if selected_person_name and selected_profession_title:
                selected_profession = next(profession for profession in all_professions if
                                           profession[language_key] == selected_profession_title)

                person_skills = self.db_manager.get_person_skills(selected_person_name)

                person_skill_titles = {skill[language_key] for skill in person_skills}

                profession_skills = self.db_manager.get_skills_connected_to_profession(selected_profession['title'])
                profession_skill_titles = {skill[language_key] for skill in profession_skills}

                missing_skills = [skill for skill in profession_skills if skill not in person_skills]
                missing_skill_titles = [skill[language_key] for skill in missing_skills]

                if missing_skills:
                    st.subheader(labels["skills_not_acquired_by_client"].format(
                        selected_profession=selected_profession_title,
                        selected_client_name=selected_person_name
                    ))
                    for skill in missing_skills:
                        for key in ['title', 'title_fi', 'description', 'description_fi']:
                            display_name = skill_properties.get(key, key)
                            st.write(f"**{display_name}:** {skill.get(key, 'N/A')}")

                    courses = self.db_manager.get_course_for_missing_skills(selected_person_name,
                                                                            selected_profession['title'])

                    if courses:
                        st.subheader(labels["courses_to_acquire_skills"])
                        for course in courses:
                            for key in ['title', 'title_fi', 'description', 'description_fi']:
                                display_name = course_properties.get(key, key)
                                st.write(f"**{display_name}:** {course.get(key, 'N/A')}")

                            st.write("**Program Skills:**")
                            for skill in course['skills']:
                                st.write(f"**{skill_properties['title']}:** {skill.get('title', 'N/A')}")
                                st.write(f"**{skill_properties['title_fi']}:** {skill.get('title_fi', 'N/A')}")

                            st.write("**Needed Skills:**")
                            for skill in missing_skill_titles:
                                st.write(f"**{skill}**")

                        G = nx.Graph()

                        # Add client node
                        G.add_node(selected_person_name, label='client', color='#1f77b4')

                        for skill in person_skill_titles:
                            G.add_node(skill, label='skill', color='#aec7e8')
                            G.add_edge(selected_person_name, skill, label='has skill')

                        # Add missing skills and related courses
                        for course in courses:
                            program_node_label = course[language_key]
                            G.add_node(program_node_label, label='program', color='#c7c7c7')

                            for skill in missing_skill_titles:
                                G.add_node(skill, label='skill', color='#ffbb78')
                                G.add_edge(selected_person_name, skill, label='missing skill')
                                G.add_edge(skill, program_node_label, label='offered by')

                        profession_node_label = selected_profession[language_key]
                        G.add_node(profession_node_label, label='profession', color='#9edae5')

                        for skill in profession_skill_titles:
                            G.add_edge(profession_node_label, skill, label='requires skill')

                        # Graph layout
                        pos = nx.spring_layout(G)
                        y_gap = 1.0
                        x_gap = 1.5

                        skills_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == 'skill']
                        for i, skill in enumerate(skills_nodes):
                            pos[skill] = (i * x_gap, 0)

                        client_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == 'client']
                        program_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == 'program']
                        for i, node in enumerate(client_nodes):
                            pos[node] = (i * x_gap, y_gap)
                        for i, node in enumerate(program_nodes):
                            pos[node] = (i * x_gap, y_gap + y_gap)

                        profession_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == 'profession']
                        for i, node in enumerate(profession_nodes):
                            pos[node] = (i * x_gap, -y_gap)

                        fig, ax = plt.subplots(figsize=(12, 8))
                        node_colors = [G.nodes[node]['color'] for node in G.nodes]
                        nx.draw(G, pos, with_labels=False, node_size=3000, node_color=node_colors, edge_color="gray",
                                ax=ax)

                        for node, (x, y) in pos.items():
                            ax.text(x, y, s=node, bbox=dict(facecolor='white', alpha=0.6), horizontalalignment='center',
                                    verticalalignment='center', fontsize=10, rotation=45)

                        ax.set_title('Client-Skills-Programs Graph')
                        st.pyplot(fig)

                    else:
                        st.write(labels["no_programs_for_missing_skills"])

    def manage_persons(self, language):
        labels = MENU_ITEMS[language]
        person_tabs = st.tabs(
            [labels["add_person"], labels["delete_person"], labels["update_person"],
             labels["suggest_training_program"]])

        with person_tabs[0]:
            self.add_person(language)
        with person_tabs[1]:
            self.delete_person(language)
        with person_tabs[2]:
            self.update_person(language)
        with person_tabs[3]:
            self.suggest_training_program(language)
