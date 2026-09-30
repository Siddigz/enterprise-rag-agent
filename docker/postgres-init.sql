-- Runs once on first container start. The app database (recon) is created from POSTGRES_DB.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE airflow;
