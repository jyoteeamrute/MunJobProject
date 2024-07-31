from py2neo import Graph, Relationship
from managers.embedding_manager import EmbeddingManager
from services.GPTClient import GPTClient
from managers.skill_manager import SkillManager
from managers.course_manager import CourseManager
from managers.profession_manager import ProfessionManager
from managers.person_manager import PersonManager
from config import load_env_variables, suppress_warnings
from constants import *


class DatabaseManager:
    def __init__(self):
        suppress_warnings()
        env_vars = load_env_variables()
        self.graph = Graph(env_vars["NEO4J_URI"], auth=(env_vars["NEO4J_USERNAME"], env_vars["NEO4J_PASSWORD"]))
        self.embedding_manager = EmbeddingManager()
        self.gpt_client = GPTClient()  # Define warning function if needed
        self.skill_manager = SkillManager(self.graph, self.embedding_manager, self.gpt_client)
        self.course_manager = CourseManager(self.graph, self.gpt_client, self.embedding_manager, self.skill_manager)
        self.profession_manager = ProfessionManager(self.graph, self.gpt_client, self.embedding_manager,
                                                    self.skill_manager)
        self.person_manager = PersonManager(self.graph, self.skill_manager)

    def get_all_skills(self, warnings_fn=None):
        return self.skill_manager.get_all_skills()

    def get_skills_connected_to_course(self, course_title, warnings_fn=None):
        query = f"""
        MATCH (c:{DEFAULT_COURSE_LABEL} {{title: $course_title}})-->(s)
        WHERE s:{DEFAULT_SKILL_LABEL} OR s:{DEFAULT_NEW_SKILL_LABEL}
        RETURN s
        """
        skills = self.graph.run(query, course_title=course_title).data()

        formatted_skills = [
            {
                'title': skill['s'].get('title'),
                'title_fi': skill['s'].get('title_fi'),
                'description': skill['s'].get('description', ''),
                'description_fi': skill['s'].get('description_fi', ''),
                'source_code': skill['s'].get('source_code', ''),
                'source_id': skill['s'].get('source_id', ''),
                'type': skill['s'].get('type', '')
            }
            for skill in skills
        ]

        return formatted_skills

    def get_all_professions(self, warnings_fn=None):
        professions = self.graph.run(f"MATCH (p:{DEFAULT_PROFESSION_LABEL}) RETURN p").data()

        profession_list = [
            {
                'source_sl': profession['p'].get('source_sl', ''),
                'source_id': profession['p'].get('source_id', ''),
                'title': profession['p'].get('title'),
                'title_fi': profession['p'].get('title_fi'),
                'description': profession['p'].get('description', ''),
                'skills': profession['p'].get('skills', '')
            }
            for profession in professions
        ]

        return profession_list

    def get_description_of_related_courses(self, skill_title, skill_label, warnings_fn=None):
        query = f"""
            MATCH (s:{skill_label} {{title: $skill_title}})--(c:{DEFAULT_COURSE_LABEL})
            RETURN c.description AS description
            """
        results = self.graph.run(query, skill_title=skill_title).data()

        descriptions = [result['description'] for result in results if result['description']]
        all_descriptions = ' '.join(descriptions)
        return all_descriptions

    def get_related_nodes(self, node_label, node_title, warnings_fn=None):
        query = f"""
            MATCH (n:{node_label} {{title: $node_title}})-[r]-(m)
            WHERE m:{DEFAULT_NEW_SKILL_LABEL} OR m:{DEFAULT_SKILL_LABEL} OR m:{DEFAULT_PROFESSION_LABEL} OR m:{DEFAULT_PERSON_LABEL} OR m:{DEFAULT_COURSE_LABEL}
            RETURN m, labels(m) AS node_labels, r.type AS relationship_type
        """

        relationships = self.graph.run(query, node_title=node_title).data()

        related_nodes = {}
        for relationship in relationships:
            related_node = relationship['m']
            node_id = related_node.identity
            node_labels = relationship['node_labels']
            relationship_type = relationship['relationship_type']

            node_info = {key: value for key, value in dict(related_node).items()}
            node_info['labels'] = node_labels
            node_info['relationship_type'] = relationship_type  # Add the relationship type to node_info

            related_nodes[node_id] = node_info

        return related_nodes

    def get_description_of_related_professions(self, skill_title, skill_label, warnings_fn=None):
        query = f"""
            MATCH (s:{skill_label} {{title: $skill_title}})--(p:{DEFAULT_PROFESSION_LABEL})
            RETURN p.description AS description
            """
        results = self.graph.run(query, skill_title=skill_title).data()

        descriptions = [result['description'] for result in results if result['description']]
        all_descriptions = ' '.join(descriptions)
        return all_descriptions

    def connect_course_to_skills(self, course_title, skills_titles, warnings_fn=None):
        try:
            for skill_title in skills_titles:
                query = """
                MATCH (c:{course_label} {{title: $course_title}})
                OPTIONAL MATCH (s1:{skill_label1} {{title: $skill_title}})
                OPTIONAL MATCH (s2:{skill_label2} {{title: $skill_title}})
                WITH c, COALESCE(s1, s2) AS s
                WHERE s IS NOT NULL
                CREATE (c)-[:TEACHES_SKILL {{type: 'MANUAL'}}]->(s)
                """.format(course_label=DEFAULT_COURSE_LABEL, skill_label1=DEFAULT_SKILL_LABEL,
                           skill_label2=DEFAULT_NEW_SKILL_LABEL)
                self.graph.run(query, course_title=course_title, skill_title=skill_title)
            return True
        except Exception as e:
            if warnings_fn:
                warnings_fn(str(e))
            return False

    def disconnect_course_from_skills(self, course_title, skills_titles, warnings_fn=None):
        try:
            for skill_title in skills_titles:
                query = """
                MATCH (c:{course_label} {{title: $course_title}})-[r]-(s)
                WHERE (s:{skill_label1} OR s:{skill_label2}) AND s.title = $skill_title
                DELETE r
                """.format(course_label=DEFAULT_COURSE_LABEL, skill_label1=DEFAULT_SKILL_LABEL,
                           skill_label2=DEFAULT_NEW_SKILL_LABEL)
                self.graph.run(query, course_title=course_title, skill_title=skill_title)
            return True
        except Exception as e:
            if warnings_fn:
                warnings_fn(str(e))
            return False

    def get_skills_connected_to_profession(self, profession_title, warnings_fn=None):
        query = f"""
        MATCH (p:{DEFAULT_PROFESSION_LABEL} {{title: $profession_title}})--(s)
        WHERE s:{DEFAULT_NEW_SKILL_LABEL} OR s:{DEFAULT_SKILL_LABEL}
        RETURN s
        """
        skills = self.graph.run(query, profession_title=profession_title).data()

        formatted_skills = [
            {
                'title': skill['s'].get('title'),
                'title_fi': skill['s'].get('title_fi'),
                'description': skill['s'].get('description', ''),
                'description_fi': skill['s'].get('description_fi', ''),
                'source_code': skill['s'].get('source_code', ''),
                'source_id': skill['s'].get('source_id', ''),
                'type': skill['s'].get('type', '')
            }
            for skill in skills
        ]

        return formatted_skills

    def connect_profession_to_skills(self, profession_title, skills_titles, warnings_fn=None):
        try:
            for skill_title in skills_titles:
                query = """
                MATCH (p:{profession_label} {{title: $profession_title}})
                OPTIONAL MATCH (s1:{skill_label1} {{title: $skill_title}})
                OPTIONAL MATCH (s2:{skill_label2} {{title: $skill_title}})
                WITH p, COALESCE(s1, s2) AS s
                WHERE s IS NOT NULL
                CREATE (p)-[:REQUIRES_SKILL {{type: 'MANUAL'}}]->(s)
                """.format(profession_label=DEFAULT_PROFESSION_LABEL, skill_label1=DEFAULT_SKILL_LABEL,
                           skill_label2=DEFAULT_NEW_SKILL_LABEL)
                self.graph.run(query, profession_title=profession_title, skill_title=skill_title)
            return True
        except Exception as e:
            warnings_fn(str(e))
            return False

    def disconnect_profession_from_skills(self, profession_title, skills_titles, warnings_fn=None):
        try:
            for skill_title in skills_titles:
                query = """
                MATCH (p:{profession_label} {{title: $profession_title}})-[r]-(s)
                WHERE (s:{skill_label1} OR s:{skill_label2}) AND s.title = $skill_title
                DELETE r
                """.format(profession_label=DEFAULT_PROFESSION_LABEL, skill_label1=DEFAULT_SKILL_LABEL,
                           skill_label2=DEFAULT_NEW_SKILL_LABEL)
                self.graph.run(query, profession_title=profession_title, skill_title=skill_title)
            return True
        except Exception as e:
            if warnings_fn:
                warnings_fn(str(e))
            return False

    def get_person_skills(self, client_name, warnings_fn=None):
        query = f"""
        MATCH (c:{DEFAULT_PERSON_LABEL} {{name: $client_name}})--(s)
        WHERE s:{DEFAULT_NEW_SKILL_LABEL} OR s:{DEFAULT_SKILL_LABEL}
        RETURN s
        """
        skills = self.graph.run(query, client_name=client_name).data()

        formatted_skills = [
            {
                'title': skill['s'].get('title'),
                'title_fi': skill['s'].get('title_fi'),
                'description': skill['s'].get('description', ''),
                'description_fi': skill['s'].get('description_fi', ''),
                'source_code': skill['s'].get('source_code', ''),
                'source_id': skill['s'].get('source_id', ''),
                'type': skill['s'].get('type', '')
            }
            for skill in skills
        ]

        return formatted_skills

    def get_course_for_missing_skills(self, client_name, profession_title, warnings_fn=None):
        query = f"""
        MATCH (client:{DEFAULT_PERSON_LABEL} {{name: $client_name}})--(clientSkill)
        WITH client, collect(clientSkill.title) AS clientSkills, collect(clientSkill) AS clientSkillNodes

        MATCH (profession:{DEFAULT_PROFESSION_LABEL} {{title: $profession_title}})--(professionSkill)
        WHERE NOT professionSkill.title IN clientSkills
        WITH client, clientSkillNodes, profession, professionSkill

        MATCH (program:{DEFAULT_COURSE_LABEL})--(professionSkill)
        WITH client, clientSkillNodes, profession, professionSkill, program, collect(professionSkill) AS skills

        CALL apoc.create.vRelationship(client, 'NEEDS_SKILL', {{}}, professionSkill) YIELD rel AS virtualRel
        RETURN client, clientSkillNodes, profession, professionSkill, program, skills, virtualRel
        ORDER BY program.title
        """
        programs = self.graph.run(query, client_name=client_name, profession_title=profession_title).data()

        formatted_programs = []
        for record in programs:
            program_node = record['program']
            skills = record.get('skills', [])
            course_node = {
                'title': program_node['title'],
                'title_fi': program_node.get('title_fi'),
                'description': program_node['description'],
                'description_fi': program_node.get('description_fi', ''),
                'skills': [
                    {
                        'title': skill.get('title'),
                        'title_fi': skill.get('title_fi')
                    } for skill in skills
                ],
                'source_code': program_node.get('source_code', ''),
                'location': program_node.get('location', '')
            }
            formatted_programs.append(course_node)

        return formatted_programs

    def connect_person_to_skills(self, person_name, skill_titles, warnings_fn=None):
        try:
            person_node = self.graph.evaluate(
                "MATCH (p:{person_label} {{name: $name}}) RETURN p".format(person_label=DEFAULT_PERSON_LABEL),
                name=person_name)
            if not person_node:
                raise ValueError(f"Person '{person_name}' not found")

            for skill_title in skill_titles:
                skill_node = self.graph.evaluate(
                    "MATCH (s:{skill_label} {{title: $title}}) RETURN s".format(skill_label=DEFAULT_SKILL_LABEL),
                    title=skill_title)
                if not skill_node:
                    skill_node = self.graph.evaluate("MATCH (s:{new_skill_label} {{title: $title}}) RETURN s".format(
                        new_skill_label=DEFAULT_NEW_SKILL_LABEL), title=skill_title)
                if not skill_node:
                    raise ValueError(f"Skill '{skill_title}' not found")

                self.graph.create(Relationship(person_node, "HAS_SKILL", skill_node))
            return True
        except Exception as e:
            if warnings_fn:
                warnings_fn(str(e))
            return False

    def disconnect_person_from_skills(self, person_name, skill_titles, warnings_fn=None):
        try:
            person_node = self.graph.evaluate(
                "MATCH (p:{person_label} {{name: $name}}) RETURN p".format(person_label=DEFAULT_PERSON_LABEL),
                name=person_name)
            if not person_node:
                raise ValueError(f"Person '{person_name}' not found")

            for skill_title in skill_titles:
                skill_node = self.graph.evaluate(
                    "MATCH (s:{skill_label} {{title: $title}}) RETURN s".format(skill_label=DEFAULT_SKILL_LABEL),
                    title=skill_title)
                if not skill_node:
                    skill_node = self.graph.evaluate("MATCH (s:{new_skill_label} {{title: $title}}) RETURN s".format(
                        new_skill_label=DEFAULT_NEW_SKILL_LABEL), title=skill_title)
                if not skill_node:
                    raise ValueError(f"Skill '{skill_title}' not found")

                self.graph.run("MATCH (p)-[r]-(s) WHERE id(p)=$person_id AND id(s)=$skill_id DELETE r",
                               person_id=person_node.identity, skill_id=skill_node.identity)
            return True
        except Exception as e:
            if warnings_fn:
                warnings_fn(str(e))
            return False
