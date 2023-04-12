# import libraries
import numpy as np
import chaospy

# define parameter ranges
fin_front_top_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_front_bottom_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_back_top_cut_angle_ranges = (0.0, np.pi / 6.0)
fin_back_bottom_cut_angle_ranges = (0.0, np.pi / 6.0)

# generate samples
samples = chaospy.generate_samples(
    order=30,
    domain=np.array(
        [
            fin_front_top_cut_angle_ranges,
            fin_front_bottom_cut_angle_ranges,
            fin_back_top_cut_angle_ranges,
            fin_back_bottom_cut_angle_ranges,
        ]
    ).T,
    rule="halton",
)
samples = samples.T
np.random.shuffle(samples)
np.savetxt("samples.txt", samples)
