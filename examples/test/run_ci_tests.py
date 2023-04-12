from tflogs_reader import check_validation_error, plot_results

if __name__ == "__main__":
    check_validation_error(
        "../helmholtz/outputs/helmholtz/",
        threshold=0.3,
        save_path="./checks/helmholtz/",
    )
    check_validation_error(
        "../discontinuous_galerkin/dg/outputs/dg/",
        threshold=0.3,
        save_path="./checks/dg/",
    )
    check_validation_error(
        "../anti_derivative/outputs/physics_informed/",
        threshold=0.3,
        save_path="./checks/physics_informed/",
    )
