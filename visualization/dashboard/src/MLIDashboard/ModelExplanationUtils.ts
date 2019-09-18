export class ModelExplanationUtils {
    public static getSortIndices(input: number[]): number[] {
        const augmented = input.map((val, index) => [val, index]);
        augmented.sort((a, b) => {
            return a[0] - b[0];
        });
        return augmented.map(augmentedVal => augmentedVal[1]);
    }

    public static buildSortedVector(matrix: number[][], selectedClassIndex?: number): number[] {
        // if no index provided, sort by absolute aggregated values
        if (selectedClassIndex === undefined) {
            return ModelExplanationUtils.getSortIndices(matrix
                .map(classArray => classArray.reduce((a, b) => a + Math.abs(b), 0)));
        }
        return ModelExplanationUtils.getSortIndices(matrix
                .map(classArray => Math.abs(classArray[selectedClassIndex])));
    }

    public static transpose2DArray<T>(matrix: T[][]): T[][] {
        return matrix[0].map((unused, index) => matrix.map(row => row[index]));
    }

    // Take the L1 norm acroos the top index of a 3d array
    public static absoluteAverageTensor(input: number[][][]): number[][] {
        const rowLength = input.length;
        const featureLength = input[0].length;
        const classLength = input[0][0].length;
        return input.reduce((a, b) => {
            return a.map((a_classArray, featureIndex) => {
                return a_classArray.map((a_cell, classIndex) => {
                    return a_cell + Math.abs(b[featureIndex][classIndex]);
                });
            });
        }, Array(featureLength).fill(0).map(val => Array(classLength).fill(0)))
        .map(classArray => classArray.map(value => value/rowLength));
    }
}
