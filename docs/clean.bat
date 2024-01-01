pushd interpret
for %%f in (*.ipynb) do (
    jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True "%%f"
)
popd