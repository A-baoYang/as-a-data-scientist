from google.cloud import storage
import os
from io import StringIO
import google_storage #travis


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/jovyan/.config/gcloud/application_default_credentials.json"
bq_project = 'gothic-province-823'
bucket_name = 'tagtoo-bigquery-export'
source_file_name = 'test.csv'
destination_blob_name = f'ECAnalysis/upload/{source_file_name}'


def upload_blob(bucket_name, source_file_name, destination_blob_name, bq_project):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=bq_project)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

    
def upload_StringIO(df, bq_project, bq_bucket='tagtoo-bigquery-export', bq_filename='abao_upload_test.csv'):
    f = StringIO()
    df.to_csv(f)
    f.seek(0)
    gcs = storage.Client(project=bq_project)
    gcs.get_bucket(bq_bucket).blob(bq_filename).upload_from_file(f, content_type='text/csv')
    
    print('File {} uploaded to {}.'.format(
        bq_filename,
        bq_bucket))

    
def bq_upload_travis(bq_project, localfile='event_before_buy_count.csv', bq_path='gs://tagtoo-bigquery-export/ECAnalysis/'):
    GSIO = google_storage.GoogleStorageIO(project=bq_project)
    blob = GSIO.upload_file(gsuri=bq_path+localfile, localpath=localfile)
    blob.make_public()
    
    print('File {} uploaded to {}.'.format(
        localfile,
        bq_path))
    
