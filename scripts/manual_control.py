import numpy as np

action = np.array([
                0, # Front right Hip
                0, # Front right Shoulder
                0, # Front right Knee
                0, # Front left Hip
                0, # Front left Shoulder
                0, # Front left Knee
                0, # Bottom right Hip
                0, # Bottom right Shoulder
                0, # Bottom right Knee
                0, # Bottom left Hip
                0, # Bottom left Shoulder
                0, # Bottom left Knee
            ])

np.save("./action.npy", action)