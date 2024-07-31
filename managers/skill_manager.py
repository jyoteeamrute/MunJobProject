import pandas as pd
from py2neo import Node, NodeMatcher
from constants import *


class SkillManager:
    def __init__(self, graph, embedding_manager, gpt_client):
        self.graph = graph
        self.embedding_manager = embedding_manager
        self.gpt_client = gpt_client
        self.matcher = NodeMatcher(graph)

    def add_skill(self, skill_source_id, skill_source_code, skill_title, skill_title_fi, skill_description,
                  skill_description_fi, skill_label=DEFAULT_SKILL_LABEL, skill_type=DEFAULT_SKILL_TYPE_PROFESSIONAL,
                  warnings_fn=None):
        # Check if the skill already exists in the graph
        existing_skill = self.graph.run(
            f"MATCH (s:{skill_label} {{title: $title}}) RETURN s",
            title=skill_title
        ).data()

        if existing_skill:
            message = f"Skill '{skill_title}' already exists in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        self.embedding_manager.add_to_embeddings(skill_title, skill_description, skill_type, warnings_fn)

        # Create and add skill node to the graph
        skill_node = Node(skill_label,
                          title=skill_title,
                          description=skill_description,
                          type=skill_type,
                          source_id=skill_source_id,
                          source_code=skill_source_code,
                          title_fi=skill_title_fi,
                          description_fi=skill_description_fi)
        self.graph.create(skill_node)
        return True

    def load_skills_from_file(self, file_path, batch_size=BATCH_SIZE, warnings_fn=None, progress_fn=None):
        skill_type_mapping = {
            'tk.skill.professional': DEFAULT_SKILL_TYPE_PROFESSIONAL,
            'tk.skill.it': DEFAULT_SKILL_TYPE_IT,
            'tk.skill.soft': DEFAULT_SKILL_TYPE_SOFT,
            'tk.skill.language': DEFAULT_SKILL_TYPE_LANGUAGE,
            'Language-skill': DEFAULT_SKILL_TYPE_LANGUAGE
        }

        try:
            # Load data from Excel file
            df = pd.read_excel(file_path)

            # Ensure necessary columns are present
            required_columns = ['ID', 'CODE', 'SKILL TYPE', 'title_en', 'desc_en', 'title_fi', 'desc_fi']
            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            # Initialize the progress bar
            total_rows = len(df)
            if progress_fn:
                progress_step = 100 / total_rows if total_rows else 1

            # Prepare batch processing
            skills_batch = []

            # Iterate over rows and add each skill to the batch
            for index, row in df.iterrows():
                # Validate title_en and title_fi are not empty
                if row['title_en'] and row['title_fi']:

                    # if skill type not found in the dictionary - type defaults as "Professional"
                    mapped_skill_type = skill_type_mapping.get(row['SKILL TYPE'], DEFAULT_SKILL_TYPE_PROFESSIONAL)

                    skill_data = {
                        'skill_source_id': row['ID'],
                        'skill_source_code': row['CODE'],
                        'skill_title': row['title_en'],
                        'skill_title_fi': row['title_fi'],
                        'skill_description': row['desc_en'],
                        'skill_description_fi': row['desc_fi'],
                        'skill_label': DEFAULT_SKILL_LABEL,
                        'skill_type': mapped_skill_type
                    }

                    skills_batch.append(skill_data)
                else:
                    if warnings_fn:
                        warnings_fn(f"Skipping row {index + 1}: Both 'title_en' and 'title_fi' must be non-empty.")

                # Update progress bar
                if progress_fn:
                    progress_fn(min((index + 1) * progress_step, 100))

                # When batch size is reached, process the batch
                if len(skills_batch) >= batch_size:
                    self.add_skills_batch(skills_batch, warnings_fn)
                    skills_batch.clear()  # Clear the batch

            # Process any remaining skills in the last batch
            if skills_batch:
                self.add_skills_batch(skills_batch, warnings_fn)

            print("All skills loaded from file and added successfully.")
            return True

        except Exception as e:
            print(f"Error loading skills from file: {e}")
            if warnings_fn:
                warnings_fn(f"Error loading skills from file: {e}")
            return False

    def add_skills_batch(self, skills_batch, warnings_fn=None):
        try:
            for skill in skills_batch:
                self.add_skill(
                    skill_source_id=skill['skill_source_id'],
                    skill_source_code=skill['skill_source_code'],
                    skill_title=skill['skill_title'],
                    skill_title_fi=skill['skill_title_fi'],
                    skill_description=skill['skill_description'],
                    skill_description_fi=skill['skill_description_fi'],
                    skill_label=skill['skill_label'],
                    skill_type=skill['skill_type'],
                    warnings_fn=warnings_fn
                )
        except Exception as e:
            message = f"An error occurred while adding skills batch: {e}"
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

    def delete_skill(self, skill_title, warnings_fn=None):
        # Try to find the skill in the graph with different labels
        existing_skill = self.matcher.match(DEFAULT_SKILL_LABEL, title=skill_title).first()
        if not existing_skill:
            existing_skill = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=skill_title).first()

        if not existing_skill:
            message = f"Skill '{skill_title}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        title = existing_skill['title']
        description = existing_skill['description']
        type = existing_skill['type']

        # Delete the skill node from the graph
        self.graph.delete(existing_skill)
        print(f"Skill '{skill_title}' deleted from the graph.")

        self.embedding_manager.delete_from_embeddings(title, description, type, warnings_fn)
        return True

    def update_skill(self, old_skill_title, warnings_fn=None, **kwargs):
        # Attempt to find the existing skill in the graph
        existing_skill = self.matcher.match(DEFAULT_SKILL_LABEL, title=old_skill_title).first()
        if not existing_skill:
            existing_skill = self.matcher.match(DEFAULT_NEW_SKILL_LABEL, title=old_skill_title).first()

        # If the skill does not exist, print a warning and return False
        if not existing_skill:
            message = f"Skill '{old_skill_title}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        # Store old skill properties for embedding manager update
        old_skill_type = existing_skill['type']
        old_skill_title = existing_skill['title']
        old_skill_description = existing_skill['description']

        # Determine new skill type, if provided, otherwise retain the old type
        new_skill_type = kwargs.get('type', old_skill_type)

        # Update existing skill properties with new values provided in kwargs
        updated_properties = kwargs
        for key, value in updated_properties.items():
            existing_skill[key] = value

        # Push the updated skill back to the graph database
        self.graph.push(existing_skill)
        print(f"Skill '{old_skill_title}' updated with new properties.")

        # If title, description, or type has changed, update the embeddings
        if 'title' in updated_properties or 'description' in updated_properties or old_skill_type != new_skill_type:
            new_skill_title = existing_skill.get('title', '')
            new_skill_description = existing_skill.get('description', '')

            self.embedding_manager.update_embeddings(
                old_skill_title, old_skill_description,
                new_skill_title, new_skill_description,
                old_skill_type, new_skill_type,
                warnings_fn
            )

        return True

    def confirm_skill(self, skill_title, warnings_fn=None):
        query = f"MATCH (s:{DEFAULT_NEW_SKILL_LABEL} {{title: $title}}) RETURN s"
        existing_skill = self.graph.run(query, title=skill_title).evaluate()

        if not existing_skill:
            message = f"Skill '{skill_title}' with label '{DEFAULT_NEW_SKILL_LABEL}' does not exist in the graph."
            print(message)
            if warnings_fn:
                warnings_fn(message)
            return False

        self.graph.run(
            f"MATCH (s:{DEFAULT_NEW_SKILL_LABEL} {{title: $title}}) REMOVE s:{DEFAULT_NEW_SKILL_LABEL} SET s:{DEFAULT_SKILL_LABEL}",
            title=skill_title)
        print(f"Skill '{skill_title}' label changed from '{DEFAULT_NEW_SKILL_LABEL}' to '{DEFAULT_SKILL_LABEL}'")
        return True

    def handle_new_skill(self, extracted_skill, warnings_fn=None):
        similar_skills = self.embedding_manager.find_top_similar_skills(extracted_skill, " ", top_similar=1)
        if similar_skills and similar_skills[0][1] > DEFAULT_THRESHOLD_FOR_CREATING_NEW_SKILLS:
            matched_skill_title, similarity_score = similar_skills[0]
            relationship_type = f"SM [{extracted_skill}] {similarity_score:.2f}"
        else:
            self.add_skill(
                skill_source_id="none",
                skill_source_code="none",
                skill_title=extracted_skill,
                skill_title_fi="none",
                skill_description="none",
                skill_description_fi="none",
                skill_label=DEFAULT_NEW_SKILL_LABEL,
                skill_type=DEFAULT_SKILL_TYPE_PROFESSIONAL,
                warnings_fn=warnings_fn
            )
            matched_skill_title = extracted_skill
            relationship_type = "NEW"

        return matched_skill_title, relationship_type

    def get_all_skills(self):
        skills_original = self.matcher.match(DEFAULT_SKILL_LABEL).all()
        skills_new = self.matcher.match(DEFAULT_NEW_SKILL_LABEL).all()
        all_skills = skills_original + skills_new

        return [
            {
                'title': skill.get('title'),
                'title_fi': skill.get('title_fi'),
                'description': skill.get('description', ''),
                'description_fi': skill.get('description_fi', ''),
                'source_code': skill.get('source_code', ''),
                'source_id': skill.get('source_id', ''),
                'type': skill.get('type', '')
            }
            for skill in all_skills
        ]

    def get_all_new_skills(self):
        skills_new = self.matcher.match(DEFAULT_NEW_SKILL_LABEL).all()

        return [
            {
                'title': skill.get('title'),
                'title_fi': skill.get('title_fi'),
                'description': skill.get('description', ''),
                'description_fi': skill.get('description_fi', ''),
                'source_code': skill.get('source_code', ''),
                'source_id': skill.get('source_id', ''),
                'type': skill.get('type', '')
            }
            for skill in skills_new
        ]
