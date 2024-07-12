pushd interpret
for %%f in (*.ipynb) do (
    jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True "%%f"
)
popd
pushd interpret\python\api
for %%f in (*.ipynb) do (
    jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True "%%f"
)
popd
pushd interpret\python\examples
for %%f in (*.ipynb) do (
    jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True "%%f"
)
popd
pushd benchmarks
for %%f in (*.ipynb) do (
    jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True "%%f"
)
popd
