# Data Engineering II Project
## Evaluating the accuracy of prediction of stargazers in open source projects (Project 3)
Desription:
With the advent of social media, the idea of popularity is used to identify who is performing better
than the others. For example, YouTube subscribers can be used as a direct measure for a
channel's popularity. Similarly, GitHub users show their appreciation for a project by putting
stargazer (or star) to the project. Therefore, the number of stargazers indicate the popularity of a
GitHub project.
In this mini-research project, our task is to study which prediction model has the highest accuracy
in predicting the number of stars for a GitHub repository. For this task, several steps are required
to complete this project.

## Project Structure:
```text
├── src
│   ├── prod (The production/user facing application)
│   │   ├── app.py
│   │   ├── ...
│   │   ├── compose.yml
│   │   └── Dockerfile
│   ├── dev (The training cluster/model development application)
│   │   ├── run_training.py
│   │   ├── evaluate.py
│   │   ├── models
│   │   │     ├── my_model.x
│   │   │     └── your_model.x
│   │   └── Dockerfile
│   ├── deploy (Infastructure related code for running the apps)
│   │   ├── configuration.yml
│   │   ├── dev-cloud.cfg
│   │   ├── prod-cloud.cfg
│   │   ├── start_instance.py
│   │   └── client-cloud-config.txt
```
## Instructions to extract data from GitHub API
- Clone the GitHub repository
- Create a JSON file with a GitHub authentication token in it.
- Paste the git token in the below format :
``` JSON
     {"github_token": "GitHub Token"}
```
- Run the code using the below command :
``` shell
    python3 "data_extraction.py"
```
Note: The dataset extracted repositoreis.json is in dev server.

## Deployment server setup :
Deployment server can be setup in local VM or by creating a seperate instance in SSC dashboard.

Note: 
- Replace the <github_token> with your github personal access token.
- The packages and the cluster SSH key needed for openstackAPI environment and ansible setup is mentioned in requirements.txt and bash script. 
- Client tools and API for openstack is available in https://docs.openstack.org/install-guide/environment-packages-ubuntu.html 

### In local VM :

Please run the shell script as below:
``` shell
      sudo bash requirements.sh
```


### VM from SSC dashboard:
- Please upload requirement.txt in configuration step.

## Configuration steps for deploy server
- Go to the de_2_project/src/deploy. This directory contains the following files used to contextualize the production and development servers.
    - client-cloud-config.txt
    - configuration.yml
    - dev-cloud-cfg.txt
    - prod-cloud-cfg.txt
    - start-instance.py
- To run the script start-instance.py in openstackAPI environment follow the below steps :
    - Download the Runtime Configuration (RC) file (version 3) from the SSC site (Top left frame, Project->API Access->Download OpenStack RC File).
    - Set API access password. Goto https://cloud.snic.se/, Left frame, under Services "Set your API password".
    - Check if the RC file has the following enviroment variables:
  ``` shell
      export OS_USER_DOMAIN_NAME="snic"
      export OS_IDENTITY_API_VERSION="3"
      export OS_PROJECT_DOMAIN_NAME="snic" 
      export OS_PROJECT_NAME="SNIC 2021/18-43"
  ```
 - Set the environment variables by sourcing the RC-file:
   
   ``` shell
           source <project_name>_openrc.sh
   ```
  NOTE: You need to enter the API access password.
  
  - Path in which cluster key should be saved /home/ubuntu/cluster-keys/RSA-key
  - The cluster ssh keys is at the following location:
       - Private key: /home/ubuntu/cluster-keys/RSA-key
       - public key: /home/ubuntu/cluster-keys/RSA-key.pub

- Go to de_2_project/src/deploy/ open prod-cloud-cfg.txt delete the old key from the section ssh_authorized_keys: and copy the complete contents of /home/ubuntu/cluster-keys/cluster-key.pub in the prod-cloud-cfg.txt file.

  Repeat same step as above with the dev-cloud-cfg.txt. Delete the old key from the section ssh_authorized_keys: and copy the complete contents of /home/ubuntu/cluster-keys/cluster-key.pub in the dev-cloud-cfg.txt file.
  
- Run the start_instance.py code.
 ``` shell
         python3 start_instance.py
 ```
   Two servers will be created automatically as dev and prod and will show up in the SNIC cloud. 
   
   The output will give the internal IP's of the VMs. 


- Switch back to user ubuntu Login to production and dev server to check the access permissions using below commands:

``` shell
ssh dev
```

Repeat the same for the dev VM
```shell
ssh prod
```

- Switch back to deploy server
- Run Ansible script in the de_2_project/src/deploy/directory in sudo/bash mode
``` shell
export ANSIBLE_HOST_KEY_CHECKING=False
ansible-playbook configuration.yml --private-key=/home/ubuntu/cluster-keys/RSA-key
```
Wait for 10 to 15 mins to complete the contextualization process.


## Running the production app (src/prod)
#### Purpose: The production or user-facing application is a basic HTML server focused on predicting the github stars count for repositories. The app is running the best-trained model from the development server/app.
- Locally:
```bash
cd src/prod
docker compose up --build
// OR in development mode with
docker compose watch
```
- In the cloud (real production):
Starting the services and whole app.
```bash
cd src/prod
docker compose up --build
```
Scaling the task workers:
```bash
docker compose up --scale worker=<DESIRED WORKER COUNT>
```
Viewing the application (options):
* Frontend application is available at localhost:5100 on the host machine (use curl)
* Expose the application publicly by attatching a floating IP to the host machine and visiting <FLOATING IP>:5100
* Create ssh tunnel(s) from the prod server to the host client machine then finally to your computer.


## Running the ML training app (src/dev)
#### Purpose: The src/dev application contains all the code required for developing the machine learning models that get deployed to the user facing app (production). The services include a jupyter server for playing with the data and models and a ray cluster for executing the final model training. Running the train.py script will train the models and save the best one as FINAL_MODEL.py
- Local debugging:
Note: Ray images for docker do not have bindings for apple silicon. Ray will fail on apple chips.
```bash
cd de_2_project/src/dev
docker compose watch
```
Then to run the training or play with the scripts exec into the head container and run the train.py script using ray
```bash
docker exec -it <RAY-HEAD-CONTAINER-ID> /bin/bash
ray job submit -- python train.py
```

- In the cloud:
Start the ray cluster if it is not running:
```bash
cd de_2_project/dev
docker compose up --build
```
To run a new training and save the model:
```bash
docker exec -it <RAY-HEAD-CONTAINER-ID> /bin/bash
ray job submit -- python train.py
```
To scale the ray cluster:
```bash
docker compose up --scale ray-worker=<WORKER COUNT (number)>
```
The training results will be written to a docker volume, accessible from host at:
```bash
/var/lib/docker/volumes/dev_shared/_data/FINAL_MODEL.pickle
```
It can be deployed to the production server by staging it in the git repo at:
```bash
/home/ubuntu/de_2_project/src/dev/publish
```
And then, after committing, pushing it to the pre-configured remote using:
```
git push production
```
