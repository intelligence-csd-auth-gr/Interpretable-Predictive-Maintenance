## VisioRed Demonstration

In this directory, instructions for building and running a docker in order to try VisioRed on your browser, are presented.

## Instructions
Please ensure you have docker installed on your desktop. Then navigate to this subfolder on your terminal:
```bash
docker build -t visioreddemo .
```
After succesfully installing VisioRed, please do:
```bash
docker run -p 8866:8866 visioreddemo
```

Then, in your terminal copy the localhost url and open it in your browser.

Otherwise, you can directly pull the dockerized app through your terminal via:

```bash
docker pull pspyrido/visiored
```
After succesfully installing VisioRed, please do:
```bash
docker run -p 8866:8866 pspyrido/visiored
```


## Contributors on VisioRed
Name | Email
--- | ---
Spyridon Paraschos | pspyrido@csd.auth.gr
Ioannis Mollas | iamollas@csd.auth.gr
Grigorios Tsoumakas | greg@csd.auth.gr
Nick Bassiliades | nbassili@csd.auth.gr
