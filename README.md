# pawan
PArticle Wake ANalysis

The main branch comprises of VPM code that runs on the CPU. The GPU-capable code is on the 'gpu-integration' branch and utilises CUDA for code acceleration. In addition, it utilises the tecplot library to output szplt files. CPU code, on the other hand, utilises the pytecplot package in python for post-processing in order to generate *plt files.

The vortex ring cases in either branch can be used to verification purposes. In order to test/use the coupling functionalities, the rotorcraft aeromechanics solver Dymore is required (not part of this repo).

## [Vortex ring self-induced convection](https://thesis.library.caltech.edu/4385/)
<p float="left">
  <img src="https://github.com/user-attachments/assets/ecc4ffeb-e204-4299-99fc-aa238c0decc7" width="250" /> 
  <img src="https://github.com/user-attachments/assets/28934a1c-f108-42f7-88b4-6a48d780aa3e" width="250" />
</p>

## Static wing
<img src="https://github.com/kumar-sumeet/pawan/assets/74828659/2cff1080-584a-41af-9fc8-b90c436e5913" align="middle" width="250" >

## Pitching/morphing 2D 'wing'
<p float="left">
  <img src="https://github.com/user-attachments/assets/970ab2fb-f2d5-4871-9c8b-2c7111824b6d" width="250" /> 
  <img src="https://github.com/user-attachments/assets/7ef58781-5fdc-48c2-8059-f490d5da7229" width="250" />
</p>
<p float="left">
  <img src="https://github.com/user-attachments/assets/1d003974-18f7-470c-9928-d22b628f83e9" width="250" /> 
  <img src="https://github.com/user-attachments/assets/0308c6b1-84bc-4f67-8e1e-507340a0a1a6" width="250" />
</p>

## Pitching finite-wing
<p float="left">
  <img src="https://github.com/user-attachments/assets/67de5359-4f83-4d5c-bd18-aa20a2cf0667" width="250" /> 
  <img src="https://github.com/user-attachments/assets/4b0c34d3-c8fb-42f0-9649-75d7db24045b" width="250" /> 
</p>

## [HART II rotor](https://www.dlr.de/en/site/hart-ii/about-hart-ii)
<p float="left">
  <img src="https://github.com/user-attachments/assets/2d7bff7d-fe9b-40fe-959a-7824f85d5375" width="250" />
  <img src="https://github.com/user-attachments/assets/12487401-ea4b-4b38-9a2f-0df5ea21796c" width="250" />
</p>

## [UT Austin rotors](https://www.sciencedirect.com/science/article/abs/pii/S1270963821003576)
<p float="left">
  <img src="https://github.com/user-attachments/assets/9ebe2ba7-8fe0-4c48-9068-f91ef5788a03" width="250" /> 
  <img src="https://github.com/user-attachments/assets/adaea039-9d44-4af0-80d9-a3f05de4bb41" width="250" />
  <img src="https://github.com/user-attachments/assets/d7cce60d-5f68-49da-aae8-17ec8c120455" width="250" />
</p>

## [Bo 105 rotor](https://ntrs.nasa.gov/api/citations/20205003457/downloads/Jacklin%20TP-20205003457_Vol%20I_Final_7-13-2020.pdf)
<p float="left">
  <img src="https://github.com/user-attachments/assets/87d821ac-0cd5-49e5-abf0-8436856424ca" width="250" />
  <img src="https://github.com/user-attachments/assets/c394eb4e-f5f3-471f-a4c7-b4ea993e7212" width="250" /> 
</p>
<p float="left">
  <img src="https://github.com/user-attachments/assets/2a439577-9862-4f5d-91dd-3d4e2f8c6761" width="250" /> 
  <img src="https://github.com/user-attachments/assets/21eaa0ef-46e1-491c-adb0-834d1050b208" width="250" />
</p>


## Installation


### Linux:
```c
$ git clone 
$ cd pawan
$ mkdir build && cd build 
$ rm -rf *
$ cmake ..
$ make 
```
