# Task-Transfer

Test how good LVLMs are in visual in-context learning

## Task2task pairs

| Task | colorization | deblurring | dehazing | demoireing | denoising | deraining | harmonization | inpainting | light enhancement | reflection removal | shadow removal | style transfer |
|:----:|:------------:|:----------:|:--------:|:----------:|:---------:|:---------:|:-------------:|:----------:|:-----------------:|:------------------:|:--------------:|:--------------:|
| **colorization**       | - |   |   |   |   |   | ✔ |   |   |   |   | ✔ |
| **deblurring**         |   | - | ✔ | ✔ |   | ✔ |   |   |   |   |   |   |
| **dehazing**           |   |   | - |   | ✔ | ✔ |   |   |   |   |   |   |
| **demoireing**         |   |   | ✔ | - |   |   |   |   |   |   |   |   |
| **denoising**          |   | ✔ |   |   | - |   |   |   | ✔ |   |   |   |
| **deraining**          |   |   |   | ✔ | ✔ | - |   |   |   |   |   | ✔ |
| **harmonization**      |   |   |   |   |   |   | - |   | ✔ |   |   | ✔ |
| **inpainting**         | ✔ |   |   |   |   |   | ✔ | - | ✔ |   |   | ✔ |
| **light enhancement**  | ✔ |   |   |   |   | ✔ |   |   | - |   | ✔ |   |
| **reflection removal** |   |   | ✔ |   |   |   |   |   |   | - |   |   |
| **shadow removal**     |   |   |   |   |   | ✔ |   |   |   | ✔ | - |   |
| **style transfer**     |   |   |   |   |   |   |   |   | ✔ |   |   | - |
