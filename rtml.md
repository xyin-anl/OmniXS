---
  pdf_document: null
  geometry: margin=2cm
  output: pdf_document
---

# Summary of my role in Real-Time Machine Learning (RTML) project

Goal of this project is to develop a model for edge processing of particle data with focus on benchmarking difference in EDGE Hardware and FPGA.

## Quarter 1: (Reproducing ATLAS model, Data)

- **Goal**: Replicate the model performance of the ATLAS model
- **Issue**: The performance of our model seemed much better than the ATLAS
  model. It needed to be verified if it was too good to be true.
- **Action**: Went through different model architectures of paper and parsed
  through codebase of ATLAS model (LCStudies)
- **Results**: I found some inconsistencies (missing layers) in the LCStudies
  codebase that might have affected the performance of the model developed by
  ATLAS team previously.

- **New Developments**:

  - We learned we cannot publish using the data that we have been doing the
    model studies.
  - It was decided to try the new open [dataset](http://opendata.cern.ch/record/15012)

- **New Direction**
  - Since we will be working on new dataset now, we decided to not spend more
    time debugging the ATLAS code.

## Quarter 2: (Model development pipeline)

- **Goal**: Develop a model pipeline.

- **Actions**:

  - **New ideas for architecture**: While waiting for data, I went through
    literature for multiple ways that we could use to increase the model
    performance, including the use of symmetry in data to our advantage and 3D
    these approaches.

  - **NAS**: Started developing a versatile NAS framework for model development
    that would be easily adjustable to fit the needs of new data.

- **Results**: Completed the NAS framework for adjustable model specification
  using Optuna and lightning integration. This would allow flexible changes to
  model depending on the new data.

- **New Developments**:

  - **Possible Submission**:We considered submitting for EDGE2023 (deadline march
    5th) but it seems it is infeasible given both the lack of data and FPGA experts
    on team.

  - **New collaboration**: Recognizing our need for FPGA experts, we started
    collaborating with SBU teams to get help on FPGA using their own tools that
    They wanted to compare it with HLS4ML as well.

## Quarter 3: (Benchmarking pipeline)

- **Goal**: Develop benchmarking pipeline benchmarking in an edge device
- **Actions**:
  - Learned Cuda upto the point needed for completion of this task including
    attending a bootcamp on it.
  - Developed a containerized environment for benchmarking in Orin hardware
    after multiple issues with NSight systems in Orin
  - Prelims benchmarking for different power and precision showed
  - Layerwise breakdown of latency performance was done
  - Layerwise breakdown of latency performance was done
- **Results**: The TensorRT pipeline for the benchmarking model was completed

Note: Now, we started collaborating with an FPGA expert for the physics department as
well, to get help on HLS4ML from the BNL side.

## Quarter 4: (FPGA Synthesis pipeline)

- **Goal**: Collaborate with domain experts to solve issues so that the paper
  can be published on time when data becomes available.

  Note: Each action described herein involved quickly acquiring knowledge
  outside my primary domain of expertise.

- **Actions/Results**:

  - **FPGA Issue 1**: SBU collaborators faced an issue with model concat layer
    HLS synthesis that they were not able to resolve even after changing the
    model design on their own.

  - **Action/Results**: After going through the source code HLS4ML library, I
    resolved the issue faced by SBU collaborators by finding and fixing the bug
    in the open-source library. If not fixed, this issue would have required
    significant model overhaul toward non-optimal design.

  - **FPGA Issue 2**: BNL FPGA expert faced an issue with model layers not
    being compiled by HLS4ML
  - **Action/Results**: I tracked the issue to ONNX optimization and provided
    temporary fix. Expecting further future issues with ONNX, I re-implemented
    the model using Keras and provided to the FPGA experts. This would avoid
    potential issue of using ONNX in HLS4ML which did not had full support for
    it.

  - **FGPA Issue 3**: BNL FPGA expert faced an issue in the compilation of the
    model using Vivado HLS.
  - **Action/Results**: Recognizing the issue with the tricky environment
    needed for Vivado HLS, I made the dockerized environment for Vivado HLS and
    provided it to FPGA experts so that we no longer face on-going environment
    integration issues.

  - **FPGA Issue 4**: The compiled synthesis showed more LUT usage in FPGA than
    is available, as pointed out by the SBU FPGA expert.
  - **Actions/Results**:I tracked the source of the issue to the FIFO buffer
    optimization and provided a report to SBU experts on the effect of FIFO
    optimizations and Reuse factor.

## The final submission attempt

- **Summary of results produced** By now, I have developed the full pipeline:
  Model Training -> NAS using Optuna -> Performance benchmarking using TensorRT
  -> HLS synthesis using HLS4ML -> FPGA synthesis using Vivado HLS
- Considering the one-week deadline, I also converted the NAS pipeline to
  distributed processing so that we can meet the deadline when data becomes
  available.
- **Last week of FY**: The available pipeline, readied to be tested with new
  data, could not be started without data and abstract submission was not
  possible.

**END OF FY**

Note:

- All things/codes listed here were made available in Github and results were
  presented in the meetings, in the said time period.
- Recognizing the possible waste of resources and time for everyone involved, I
  went ahead with the processing of data myself with help from domain expert
  who also confirmed its validity. Paper submission is now possible if
  supported by collaborators.
