def gen(datasets, preprocessing, classifier):
    matrix = {}

    parameters = {}
    settings = {'n_fold': 20}
    exclude = {}

    exclude["dataset"] = datasets
    exclude["classifier"] = classifier

    parameters["dataset"] = datasets
    parameters["preprocessing"] = preprocessing
    parameters["classifier"] = classifier

    matrix["parameters"] = parameters
    matrix["settings"] = settings
    matrix["exclude"] = exclude

    return matrix


if __name__ == '__main__':
    print(gen([], [], []))

