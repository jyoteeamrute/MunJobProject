import pandas as pd
from py2neo import Node, Relationship, NodeMatcher
from constants import *


class CourseManager:
    def __init__(self, graph, gpt_client, embedding_manager, skill_manager):
        self.graph = graph
        self.gpt_client = gpt_client
        self.embedding_manager = embedding_manager
        self.skill_manager = skill_manager
        self.matcher = NodeMatcher(graph)

    def add_course(self, course_title_fi, course_title="", course_description="", course_description_fi="",
                   course_source_code="", course_location="", course_skills="", warnings_fn=None):

        title_description_fi = f"{course_title_fi}::{course_description_fi}"
        existing_course = self.matcher.match(DEFAULT_COURSE_LABEL, source_code=course_source_code,
                                             title_fi=course_title_fi).first()
        if not existing_course:
            existing_course = self.matcher.match(DEFAULT_COURSE_LABEL,
                                                 original_title_description=title_description_fi).first()

        if existing_course:
            message = "Course already exists in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return
        if not course_title:
            course_title = self.gpt_client.translate(course_title_fi)
        if not course_description:
            course_description = self.gpt_client.translate(course_description_fi)

        if not course_skills:
            course_skills = self.gpt_client.extract_skills_from_course_description(f"{course_title}::{course_description}")

        course_node = Node(DEFAULT_COURSE_LABEL,
                           title=course_title,
                           description=course_description,
                           skills=course_skills,
                           source_code=course_source_code,
                           location=course_location,
                           title_fi=course_title_fi,
                           description_fi=course_description_fi
                           )

        self.graph.create(course_node)

        # self.embedding_manager.add_to_embeddings(course_title, skills, 'Courses', warnings_fn=warnings_fn)

        filtered_skills = self.embedding_manager.filter_skills(course_title, course_skills,
                                                               warnings_fn=warnings_fn)

        matched_skills = self.gpt_client.match_skills_for_course(filtered_skills, f"{course_title}::{course_skills}")

        for pair in matched_skills:
            extracted_skill = pair['extracted_skill']
            matched_skill = pair['common_skill']

            if matched_skill.lower() == 'new':
                matched_skill, relationship_type = self.skill_manager.handle_new_skill(extracted_skill)
            else:
                relationship_type = f"EXCT [{extracted_skill}]"

            course_node = self.matcher.match(DEFAULT_COURSE_LABEL, title=course_title).first()
            if relationship_type == 'NEW':
                skill_node = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=matched_skill).first()
            else:
                skill_node = self.matcher.match(DEFAULT_SKILL_LABEL, title=matched_skill).first()

            if course_node and skill_node:
                rel = Relationship(course_node, "TEACHES_SKILL", skill_node, type=relationship_type)
                self.graph.create(rel)
            else:
                message = f"Node {matched_skill} or node {skill_node} not found"
                print(message)
                if warnings_fn:
                    warnings_fn(message)

    def add_courses_batch(self, courses_batch, warnings_fn=None):
        for course_data in courses_batch:
            self.add_course(
                course_title_fi=course_data['course_title_fi'],
                course_description_fi=course_data['course_description_fi'],
                course_source_code=course_data['course_source_code'],
                course_location=course_data['course_location'],
                warnings_fn=warnings_fn
            )

    def load_courses_from_file(self, file_path, batch_size=BATCH_SIZE, warnings_fn=None, progress_fn=None):

        try:
            df = pd.read_excel(file_path)

            required_columns = ['Title', 'Training code number', 'Description', 'Location city']
            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            # Initialize the progress bar
            total_rows = len(df)
            if progress_fn:
                progress_step = 100 / total_rows if total_rows else 1

            # Prepare batch processing
            courses_batch = []

            # Iterate over rows and add each course to the batch
            for index, row in df.iterrows():
                course_data = {
                    'course_title_fi': row['Title'],
                    'course_description_fi': row['Description'],
                    'course_source_code': row['Training code number'],
                    'course_location': row['Location city']
                }

                courses_batch.append(course_data)

                # Update progress bar
                if progress_fn:
                    progress_fn(min((index + 1) * progress_step, 100))

                # When batch size is reached, process the batch
                if len(courses_batch) >= batch_size:
                    self.add_courses_batch(courses_batch, warnings_fn)
                    courses_batch.clear()  # Clear the batch

            # Process any remaining courses in the last batch
            if courses_batch:
                self.add_courses_batch(courses_batch, warnings_fn)

            print("All courses loaded from file and added successfully.")
        except Exception as e:
            message = f"Error loading courses from file: {e}"
            print(message)
            if warnings_fn:
                warnings_fn(message)

    def delete_course(self, course_title, warnings_fn=None):
        course_node = self.matcher.match(DEFAULT_COURSE_LABEL, title=course_title).first()

        if not course_node:
            message = f"Course '{course_title}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return

        course_title = course_node['title']
        # course_skills = course_node['skills']

        self.graph.run("MATCH (c)-[r]-() WHERE id(c)=$id DELETE r", id=course_node.identity)
        self.graph.delete(course_node)
        # self.embedding_manager.delete_from_embeddings(course_title, course_skills, warnings_fn)

        print(f"Course '{course_title}' deleted from the graph.")

    def update_course(self, course_title, **kwargs):
        course_node = self.matcher.match(DEFAULT_COURSE_LABEL, title=course_title).first()

        if not course_node:
            message = f"Course '{course_title}' does not exist in the graph."
            print(message)
            if 'warnings_fn' in kwargs:
                kwargs['warnings_fn'](message)
            return

        # for course embedding
        # old_course_title = course_node['title']
        # old_course_skills = course_node['skills']

        updated_fields = []
        for key, value in kwargs.items():
            if key != 'warnings_fn' and value is not None:
                course_node[key] = value
                updated_fields.append(key)

        self.graph.push(course_node)

        # for course embedding
        # if 'title' in updated_fields or 'skills' in updated_fields:
        # new_course_title = course_node['title']
        # new_course_skills = course_node['skills']
        # self.embedding_manager.update_embeddings(old_course_title, old_course_skills, new_course_title, new_course_skills, 'Course', warnings_fn=None)

    def get_all_courses(self):
        courses = self.matcher.match(DEFAULT_COURSE_LABEL).all()
        return [
            {
                'id': course.identity,
                'title': course.get('title', 'N/A'),
                'title_fi': course.get('title_fi', 'N/A'),
                'description': course.get('description', 'N/A'),
                'description_fi': course.get('description_fi', 'N/A'),
                'location': course.get('location', 'N/A'),
                'source_code': course.get('source_code', 'N/A'),
                'skills': course.get('skills', 'N/A')
            }
            for course in courses
        ]
