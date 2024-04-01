# Importing necessary dependencies
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as f
from awsglue.dynamicframe import DynamicFrame

# Initializing Spark_session, glueContext and initializing Job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Specifying the source and destination location
data_clinical_trials = 's3://lilian-hospital/rawdata/clinical_trials_data.csv'
data_imaging_results = 's3://lilian-hospital/rawdata/imaging_results_data.csv'
data_lab_results = 's3://lilian-hospital/rawdata/lab_results_data.csv'
data_medical_records = 's3://lilian-hospital/rawdata/medical_records_data.csv'
data_patient_records = 's3://lilian-hospital/rawdata/patients_data.csv'
data_trial_participants = 's3://lilian-hospital/rawdata/trial_participants_data.csv'
output_dir = 's3://lilian-hospital/output/'

# Read data into a DynamicFrame 

clinical_trials = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_clinical_trials], "recurse": True},
    transformation_ctx="clinical_trials",
)

imaging_results = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_imaging_results], "recurse": True},
    transformation_ctx="imaging_results",
)

lab_results = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_lab_results], "recurse": True},
    transformation_ctx="lab_results",
)

medical_records = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_medical_records], "recurse": True},
    transformation_ctx="medical_records",
)

patient_records = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_patient_records], "recurse": True},
    transformation_ctx="patient_records",
)

trial_participants = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_trial_participants], "recurse": True},
    transformation_ctx="trial_participants",
)

# Apply the mapping based on the provided columns
mapped_clinical_trials = ApplyMapping.apply(
    frame=clinical_trials,
    mappings=[
        ("trial_id", "string", "trial_id", "string"),
        ("trial_name", "string", "trial_name", "string"),
        ("principal_investigator", "string", "principal_investigator", "string"),
        ("start_date", "string", "start_date", "string"),
        ("end_date", "string", "end_date", "string"),
        ("trial_description", "string", "trial_description", "string"),
    ],
    transformation_ctx="mapped_clinical_trials"
)

mapped_imaging_results = ApplyMapping.apply(
    frame=imaging_results,
    mappings=[
        ("result_id", "string", "result_id", "string"),
        ("patient_id", "string", "patient_id", "string"),
        ("imaging_type", "string", "imaging_type", "string"),
        ("imaging_date", "string", "imaging_date", "string"),
        ("image_url", "string", "image_url", "string"),
        ("findings", "string", "findings", "string"),
    ],
    transformation_ctx="mapped_imaging_results"
)

mapped_lab_results = ApplyMapping.apply(
    frame=lab_results,
    mappings=[
        ("result_id", "string", "result_id", "string"),
        ("patient_id", "string", "patient_id", "string"),
        ("test_name", "string", "test_name", "string"),
        ("test_date", "string", "test_date", "string"),
        ("test_result", "string", "test_result", "string"),
        ("reference_range", "string", "reference_range", "string"),
    ],
    transformation_ctx="mapped_lab_results"
)

mapped_medical_records = ApplyMapping.apply(
    frame=medical_records,
    mappings=[
        ("record_id", "string", "record_id", "string"),
        ("patient_id", "string", "patient_id", "string"),
        ("admission_date", "string", "admission_date", "string"),
        ("discharge_date", "string", "discharge_date", "string"),
        ("diagnosis", "string", "diagnosis", "string"),
        ("treatment_description", "string", "treatment_description", "string"),
        ("prescribed_medications", "string", "prescribed_medications", "string"),
    ],
    transformation_ctx="mapped_medical_records"
)

mapped_patient_records = ApplyMapping.apply(
    frame=patient_records,
    mappings=[
        ("patient_id", "string", "patient_id", "string"),
        ("first_name", "string", "first_name", "string"),
        ("last_name", "string", "last_name", "string"),
        ("date_of_birth", "string", "date_of_birth", "string"),
        ("gender", "string", "gender", "string"),
        ("ethnicity", "string", "ethnicity", "string"),
        ("address", "string", "address", "string"),
        ("contact_number", "string", "contact_number", "string"),
    ],
    transformation_ctx="mapped_patient_records"
)

mapped_trial_participants = ApplyMapping.apply(
    frame=trial_participants,
    mappings=[
        ("participant_id", "string", "participant_id", "string"),
        ("trial_id", "string", "trial_id", "string"),
        ("patient_id", "string", "patient_id", "string"),
        ("enrollment_date", "string", "enrollment_date", "string"),
        ("participant_status", "string", "participant_status", "string"),
    ],
    transformation_ctx="mapped_trial_participants"
)

# Convert to DataFrame for cleaning and tranformation
clinical_trials_df = mapped_clinical_trials.toDF()
imaging_results_df = mapped_imaging_results.toDF()
lab_results_df = mapped_lab_results.toDF()
medical_records_df = mapped_medical_records.toDF()
patient_records_df = mapped_patient_records.toDF()
trial_participants_df = mapped_trial_participants.toDF()

# Tranformation Layer
patient_dimension = patient_records_df[['patient_id', 'first_name', 'last_name', 'date_of_birth', 'gender', 'ethnicity', 'address', 'contact_number']].drop_duplicates().reset_index(drop=True)
diagnosis_dimension = medical_records_df[['diagnosis']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'diagnosis_id'})
treatment_dimension = medical_records_df[['treatment_description', 'prescribed_medications']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'treatment_id'})
imaging_type_dimension = imaging_results_df[['imaging_type']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'imaging_type_id'})
test_type_dimension = lab_results_df[['test_name']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'test_type_id'})
trial_dimension = clinical_trials_df[['trial_name', 'principal_investigator', 'start_date', 'end_date', 'trial_description']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'trial_id'})
participant_status_dimension = trial_participants_df[['participant_status']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'participant_status_id'})