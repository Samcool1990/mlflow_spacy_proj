import mlflow.spacy
import spacy
import stanza
import spacy_stanza
from spacy.language import Language
from spacy.pipeline import EntityRuler
import mlflow


import dagshub
dagshub.init(repo_owner='Samcool1990', repo_name='mlflow_spacy_proj', mlflow=True)

# Download and load models
stanza.download("en")  # Download Stanza model for English
nlp_sm = spacy.load("en_core_web_lg")  # Load SpaCy large English model
nlp_md = spacy_stanza.load_pipeline("en")  # Load Stanza pipeline into SpaCy

# Create a new blank model to combine the pipelines
nlp_combined = spacy.blank("en")

# Add pipelines from both models using string names
if nlp_sm:
    for name, component in nlp_sm.pipeline:
        if name not in nlp_combined.pipe_names:
            nlp_combined.add_pipe(name, source=nlp_sm)
if nlp_md:
    for name, component in nlp_md.pipeline:
        if name not in nlp_combined.pipe_names:
            nlp_combined.add_pipe(name, source=nlp_md)

# Ensure no duplicates and only unique components
unique_pipe_names = list(dict.fromkeys(nlp_combined.pipe_names))
nlp_combined = spacy.blank("en")
for name in unique_pipe_names:
    nlp_combined.add_pipe(name, source=nlp_sm if name in nlp_sm.pipe_names else nlp_md)

# Disable static vectors in the tok2vec component
for component_name, component in nlp_combined.pipeline:
    if component_name == "tok2vec":
        component.cfg["include_static_vectors"] = False

# Define factory functions for custom EntityRulers
@Language.factory("icd_ruler")
def create_icd_ruler(nlp, name):
    ruler = EntityRuler(nlp, validate=True)
    ruler.from_disk('./corpus/icd10cm-tabular-2024 1.jsonl')
    return ruler

@Language.factory("medra_ruler")
def create_medra_ruler(nlp, name):
    ruler = EntityRuler(nlp, validate=True)
    ruler.from_disk('./corpus/medraLLT.jsonl')
    return ruler

# Add the EntityRulers to the pipeline by their names
if "icd_ruler" not in nlp_combined.pipe_names:
    nlp_combined.add_pipe("icd_ruler", before="ner")
if "medra_ruler" not in nlp_combined.pipe_names:
    nlp_combined.add_pipe("medra_ruler", after="icd_ruler")

# Start an MLflow run with experiment management
if __name__ == "__main__":
    # Create a new mlflow experiment
    experiment_name = "spacy_stanza_combined_model_experiment"
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location="./mlruns",
            tags={"env": "dev", "version": "1.0.0"},
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # Log parameters, metrics, and model
        mlflow.log_param("iterations", 10)
        mlflow.spacy.log_model(spacy_model=nlp_combined, artifact_path="spacy_combined_model_experiment")

        print(f"Run ID: {run_id}")

    # Register the Model
    model_name = "spacy_combined_ner_model_stanza_test_experiment_model"

    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/spacy_combined_model_experiment",
            name=model_name,
        )
        print(f"Model registered as {result.name} with version {result.version}")

        # Load model by name and version
        model_uri = f"models:/{model_name}/1"
        loaded_model = mlflow.spacy.load_model(model_uri)

        # # Disable static vectors in tok2vec if necessary
        # for component in loaded_model.pipeline:
        #     if component[0] == "tok2vec":
        #         component[1].cfg["include_static_vectors"] = False

        # Disable static vectors in tok2vec if necessary
        for component_name, component in loaded_model.pipeline:
            if component_name == "tok2vec":
                component.cfg["include_static_vectors"] = False

        # Make predictions
        text = ("Alex and Ritesh are cricket players and Mariam never played baseball. "
                "Although the two local anesthetics usually do not cause methemoglobinemia, we suspect that the displacement of lidocaine "
                "from protein binding by bupivacaine, in combination with metabolic acidosis and treatment with other oxidants, "
                "was the reason for the development of methemoglobinemia.")
        doc = loaded_model(text)

        # Print entities
        for ent in doc.ents:
            print(ent.text, ent.label_)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Retrieve the MLflow experiment details
    experiment = mlflow.get_experiment(experiment_id=experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")




# import mlflow
# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)