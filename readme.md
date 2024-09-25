# Electromyography (EMG) artifact task
Last edit: 09/25/2024

## Edit history
- 09/25/2024 by Alex He - added winHandle.activate() to make sure window is on foreground
- 09/23/2024 by Alex He - upgraded to run on PsychoPy 2024.2.2
- 09/05/2024 by Alex He - removed git tracking of _lastrun.py file and added retries to pyxid2.get_xid_devices() with timeout
- 08/17/2024 by Alex He - added more print messages during c-pod connection
- 08/12/2024 by Alex He - reverted to python 3.8 as pylink connection to EyeLink does not work correctly on 3.10
- 08/04/2024 by Alex He - generated experiment scripts on python 3.10
- 08/02/2024 by Alex He - upgraded to support PsychoPy 2024.2.1
- 07/23/2024 by Alex He - upgraded to support PsychoPy 2024.2.0
- 06/30/2024 by Alex He - created finalized first draft version

## Description
This task is used to elicit a few canonical muscle artifacts as captured in EEG recording channels. By following short video clips of an experimenter demonstrating a series of facial movements, subjects will repeat the below motions:
    - lower eyebrow
    - raise eyebrow
    - squint both eyes
    - clench shut both eyes
    - wrinkle nose
    - open mouth
These facial movements, along with eye blinks, tend to introduce visible artifacts in EEG recordings. By asking subjects to intentionally produce the artifacts along with timestamps, we build a basis set of the EMG artifacts as they appear for a specific subject. Later data analysis approaches could attempt to model these EMG artifacts (such as using AR2 models with varying degrees of variances). These models can be fitted to the epochs of dedicated periods of labeled artifacts elicited in this task, and learned models can be used to detect and/or clean up artifacts in actual EEG data.

## Outcome measures
- EMG artifact signals across EEG channels
- AR2 models fitted to the timestamped epochs of motion artifacts
