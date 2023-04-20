Modulus Overview
=================

Modulus Building Blocks
-----------------------

* **Geometry and Data**
    Modulus provides both physics-informed and data-driven machine learning solutions for physics simulation problems.
    All these solutions depend on expressing the physics problem as a mathematical optimization problem.
    The mathematical optimization problem is, in turn, posed on a particular geometry and/or set of data.
    Modulus' geometry module lets users either build a geometry from scratch using primitives, or import an existing geometry from a mesh.
    For data-driven problems, Modulus has multiple methods for accessing data, including standard in-memory datasets as well as lazy loading methods for large-scale datasets.

* **Nodes**
    In Modulus, ``Node``\s represent components that will be executed in the forward pass during training.
    A ``Node`` may wrap a ``torch.nn.Module`` and provides additional information regarding its needed input and output variables.  This lets Modulus build execution graphs and automatically fill in missing components to compute required derivatives.
    ``Node``\s may contain models or functions such as PyTorch neural networks that are built into Modulus, user-defined PyTorch networks, feature transformations, or even equations.

* **Constraints**
    ``Constraint``\s are the training objectives in Modulus.
    A ``Constraint`` contains the loss function, and a set of ``Node``\s from which Modulus builds a computational graph for execution.
    Many physical problems need multiple training objectives in order for the problem to be well defined.  ``Constraint``\s provide the means for setting up such problems.

* **Domain**
    The ``Domain`` holds all ``Constraint``\s as well as additional components needed in the training process.  These additional components include ``Inferencer``\s, ``Validator``\s, and ``Monitor``\s.
    When developing in Modulus, ``Constraint``\s that the user defines are then added to the training ``Domain`` to create a collection of training objectives.

* **Solver**
    A ``Solver`` is an instance of the core Modulus trainer.  It implements the optimization loop and manages the training process.
    A ``Solver`` takes a defined ``Domain`` and calls the ``Constraint``\s, ``Inferencer``\s, ``Validator``\s, and ``Monitor``\s when required.
    During one iteration, the ``Solver`` will compute the global loss from all ``Constraint``\s and then optimize any trainable models present in the ``Node``\s provided to the ``Constraint``\s.

* **Hydra**
    Hydra is a configuration package built into Modulus.  It lets users set hyperparameters (parameters that determine the neural network's structure and govern its training) using configuration files in YAML (a standard human-readable text format).
    Hydra is the first component to be initialized when solving a problem using Modulus.  It directly influences all Modulus components.

* **Inferencers**
    An ``Inferencer`` executes just the forward pass of a set of ``Node``\s.
    ``Inferencer``\s may be used during training to assess training quantities or get predictions for visualization or deployment.
    Hydra configuration settings control the frequency at which ``Inferencer``\s are called.

* **Validators**
    ``Validator``\s work like ``Inferencer``\s, but also take validation data.
    They quantify the accuracy of the model during training,
    by validating it against physical results produced by some other method.
    ("Validation" here means the part of "verification and validation" that checks whether Modulus meets its operational requirements,
    by comparing the simulation results that Modulus computes against some "known good" result.)

* **Monitors**
    ``Monitor``\s also work like ``Inferencer``\s, but calculate specific measures instead of fields.
    These measures may be global quantities such as total energy, or local probes such as pressure in front of a bluff body.
    (A "bluff body" is a kind of shape with special fluid dynamics properties.)
    ``Monitor``\s are automatically added to Tensorboard results for viewing.
    ``Monitor`` results can also be exported to a text file in comma-separated values (CSV) format.

Modulus Development Workflow
----------------------------

The figure below illustrates a typical workflow when developing in Modulus.
Not all problems will call for exactly this workflow, but it serves as a general guide.
The key steps of this process include:

* "Load Hydra": Initialize Hydra using the Modulus ``main`` decorator to read in the YAML configuration file.
* "Load Datasets": Load data if needed.
* "Define Geometry": Define the geometry of the system if needed.
* "Create Nodes": Create any ``Node``\s required, such as the neural network model.
* "Create Domain": Create a training ``Domain`` object.
* "Create Constraint" and "Add Constraint to Domain": Create each of the :math:`N_{c}` ``Constraint``\s in turn, and add it to the ``Domain``\.
* "Create {Validator, Inferencer, Monitor}" and "Add {Validator, Inferencer, Monitor} to Domain": Create any ``Inferencer``\s, ``Validator``\s or ``Monitor``\s needed, and add them to the ``Domain``\.
* "Create Solver": Initialize a ``Solver`` with the populated training ``Domain``\.
* "Run Solver": Run the ``Solver``\.  The resulting training process optimizes the neural network to solve the physics problem.

More details of each step can be found in the :ref:`ldc` chapter which provides a hands-on introduction to Modulus.

.. _fig-modulus-dev-loop:

.. figure:: /images/user_guide/modulus_dev_workflow.png
    :alt: Modulus' training loop
    :width: 100.0%
    :align: center
    
    A typical workflow followed when developing in Modulus.


Modulus Training Algorithm
--------------------------

.. _fig-modulus-training-loop:

.. figure:: /images/user_guide/modulus_training_loop.png
    :alt: Modulus' training loop
    :width: 100.0%
    :align: center
    
    Modulus' training algorithm.
