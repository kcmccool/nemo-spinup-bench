#!/bin/bash
set -e

DATA_DIR="data"
ZIP_FILE="50.zip"
ZENODO_URL="https://zenodo.org/records/19557419/files/${ZIP_FILE}"

mkdir -p "${DATA_DIR}/50/"

echo "Downloading DINO 50-year reference data..."
curl -L -o "${ZIP_FILE}" "${ZENODO_URL}"
unzip -o "${ZIP_FILE}" -d "${DATA_DIR}/50"
rm "${ZIP_FILE}"

echo "Done. Data downloaded to ${DATA_DIR}/50/"
