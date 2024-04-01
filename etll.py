import sys
import pandas as pd
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job 
import pyspark.sql.functions as f
from awsglue.dynamicframe import DynamicFrame 
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Specifying the source and destination location
data_clinical_trials = "s3://lilian-hospital/rawdata/clinical_trials_data.csv"
data_imaging_results = "s3://lilian-hospital/rawdata/imaging_results_data.csv"
data_lab_results = "s3://lilian-hospital/rawdata/lab_results_data.csv"
data_medical_records = "s3://lilian-hospital/rawdata/medical_records_data.csv"
data_patient_records = "s3://lilian-hospital/rawdata/patients_data.csv"
data_trials_participant = "s3://lilian-hospital/rawdata/trial_participants_data.csv"
output_dir = "s3://lilian-hospital/output/"

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

trials_participant = glueContext.create_dynamic_frame.from_options(
    format_options={"quoteChar":'"', "withHeader": True, "separator":","},
    connection_type="s3",
    format="csv",
    connection_options={"paths":[data_trials_participant], "recurse": True},
    transformation_ctx="trials_participant",
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

mapped_trials_participant = ApplyMapping.apply(
    frame=trials_participant,
    mappings=[
        ("participant_id", "string", "participant_id", "string"),
        ("trial_id", "string", "trial_id", "string"),
        ("patient_id", "string", "patient_id", "string"),
        ("enrollment_date", "string", "enrollment_date", "string"),
        ("participant_status", "string", "participant_status", "string"),
    ],
    transformation_ctx="mapped_trials_participant"
)

# Convert to DataFrame for cleaning and transformation
clinical_trials_df = mapped_clinical_trials.toDF()
imaging_results_df = mapped_imaging_results.toDF()
lab_results_df = mapped_lab_results.toDF()
medical_records_df = mapped_medical_records.toDF()
patient_records_df = mapped_patient_records.toDF()
trials_participant_df = mapped_trials_participant.toDF()

# Convert to Spark DataFrame to Pandas Dataframe
clinical_trials_pd_df = clinical_trials_df.toPandas()
imaging_results_pd_df = imaging_results_df.toPandas()
lab_results_pd_df = lab_results_df.toPandas()
medical_records_pd_df = medical_records_df.toPandas()
patient_records_pd_df = patient_records_df.toPandas()
trials_participant_pd_df = trials_participant_df.toPandas()

# Transformation Layer
patient_dimension = patient_records_pd_df[['patient_id', 'first_name', 'last_name', 'date_of_birth', 'gender','ethnicity', 'address', 'contact_number']].drop_duplicates().reset_index(drop=True)
diagnosis_dimension = medical_records_pd_df[['diagnosis']].drop_duplicates().reset_index(drop=True)
diagnosis_dimension['diagnosis_id'] = range(1, len(diagnosis_dimension) + 1)
treatment_dimension = medical_records_pd_df[['treatment_description', 'prescribed_medications']].drop_duplicates().reset_index(drop=True)
treatment_dimension['treatment_id'] = range(1, len(treatment_dimension) + 1)
imaging_type_dimension = imaging_results_pd_df[['imaging_type']].drop_duplicates().reset_index(drop=True)
imaging_type_dimension['imaging_type_id'] = range(1, len(imaging_type_dimension) + 1)
test_type_dimension = lab_results_pd_df[['test_name']].drop_duplicates().reset_index(drop=True)
test_type_dimension['test_type_id'] = range(1, len(test_type_dimension) + 1)
trial_dimension = clinical_trials_pd_df[['trial_name','principal_investigator','start_date','end_date','trial_description']].drop_duplicates().reset_index(drop=True)
trial_dimension['trial_id'] = range(1, len(trial_dimension) + 1)
participant_status_dimension = trials_participant_pd_df[['participant_status']].drop_duplicates().reset_index(drop=True)
participant_status_dimension['participant_status_id'] = range(1, len(participant_status_dimension) + 1)

fact_table = medical_records_pd_df.merge(patient_records_pd_df, on='patient_id', how='inner') \
                               .merge(imaging_results_pd_df, on='patient_id', how='inner') \
                               .merge(lab_results_pd_df, on='patient_id', how='inner') \
                               .merge(trials_participant_pd_df, on='patient_id', how='inner')

fact_table = fact_table.merge(clinical_trials_pd_df, on='trial_id', how='inner')

fact_table = fact_table[['record_id', 'patient_id', 'diagnosis', 'treatment_description', 'imaging_type', 'test_name', 'test_result', 'trial_name', 'participant_status', 'admission_date', 'discharge_date', 'imaging_date', 'test_date', 'enrollment_date']]

# Convert Pandas dataframe back to Spark dataframe
patient_dimension_spark_df = spark.createDataFrame(patient_dimension)
diagnosis_dimension_spark_df = spark.createDataFrame(diagnosis_dimension)
treatment_dimension_spark_df = spark.createDataFrame(treatment_dimension)
imaging_type_dimension_spark_df = spark.createDataFrame(imaging_type_dimension)
test_type_dimension_spark_df = spark.createDataFrame(test_type_dimension)
trial_dimension_spark_df = spark.createDataFrame(trial_dimension)
participant_status_dimension_spark_df = spark.createDataFrame(participant_status_dimension)
fact_table_spark_df = spark.createDataFrame(fact_table)

# Convert back to DynamicFrame for the AWS Glue sink
patient_dimension_dyf = DynamicFrame.fromDF(patient_dimension_spark_df, glueContext, "patient_dimension_dyf")
diagnosis_dimension_dyf = DynamicFrame.fromDF(diagnosis_dimension_spark_df, glueContext, "diagnosis_dimension_dyf")
treatment_dimension_dyf = DynamicFrame.fromDF(treatment_dimension_spark_df, glueContext, "treatment_dimension_dyf")
imaging_type_dimension_dyf = DynamicFrame.fromDF(imaging_type_dimension_spark_df, glueContext, "imaging_type_dimension_dyf")
test_type_dimension_dyf = DynamicFrame.fromDF(test_type_dimension_spark_df, glueContext, "test_type_dimension_dyf")
trial_dimension_dyf = DynamicFrame.fromDF(trial_dimension_spark_df, glueContext, "trial_dimension_dyf")
participant_status_dimension_dyf = DynamicFrame.fromDF(participant_status_dimension_spark_df, glueContext, "participant_status_dimension_dyf")
fact_table_dyf = DynamicFrame.fromDF(fact_table_spark_df, glueContext, "fact_table_dyf")


# Loading layer
# Define a list of tuples containing DynamicFrames and their respective table names
dynamic_frames = [
    (patient_dimension_dyf, "patient_dimension")
]

# Define the formats and their respective configurations
formats = {
    "csv": {"format": "csv", "compression": "gzip"},
    "parquet": {"format": "glueparquet", "compression": "snappy"},
    "json": {"format": "json", "compression": None}  # JSON does not have compression option in this context
}

# Iterate through each format and DynamicFrame
for file_format, config in formats.items():
    for dyf, table_name in dynamic_frames:
        # Configure the sink based on the format
        sink = glueContext.getSink(
            path=f"{output_dir}{file_format}/",
            connection_type="s3",
            updateBehavior="UPDATE_IN_DATABASE",
            partitionKeys=[],
            compression=config["compression"],
            enableUpdateCatalog=True,
            transformation_ctx=f"{table_name}_{file_format}_sink",
        )
        sink.setCatalogInfo(
            catalogDatabase="lilian_hosp_database",
            catalogTableName=f"{table_name}_{file_format}"
        )
        sink.setFormat(config["format"])
        sink.writeFrame(dyf)

job.commit()