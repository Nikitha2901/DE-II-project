import time, os, sys, random, re
import inspect
from os import environ as env

from  novaclient import client
import keystoneclient.v3.client as ksclient
from keystoneauth1 import loading
from keystoneauth1 import session


prod_flavor = "ssc.medium" 
dev_flavor = "ssc.large"
private_net = "UPPMAX 2024/1-4 Internal IPv4 Network"
floating_ip_pool_name = "Public External IPv4 Network"
floating_ip = "130.238.28.156"
image_name = "Ubuntu 22.04 - 2024.01.15"

identifier = random.randint(1000,9999)

loader = loading.get_plugin_loader('password')

auth = loader.load_from_options(auth_url=env['OS_AUTH_URL'],
                                username=env['OS_USERNAME'],
                                password=env['OS_PASSWORD'],
                                project_name=env['OS_PROJECT_NAME'],
                                project_domain_id=env['OS_PROJECT_DOMAIN_ID'],
                                #project_id=env['OS_PROJECT_ID'],
                                user_domain_name=env['OS_USER_DOMAIN_NAME'])

sess = session.Session(auth=auth)
nova = client.Client('2.1', session=sess)
print ("user authorization completed.")

image = nova.glance.find_image(image_name)

prod_flavor = nova.flavors.find(name=prod_flavor)
dev_flavor = nova.flavors.find(name=dev_flavor)

if private_net != None:
    net = nova.neutron.find_network(private_net)
    nics = [{'net-id': net.id}]
else:
    sys.exit("private-net not defined.")

#print("Path at terminal when executing this file")
#print(os.getcwd() + "\n")
cfg_file_path =  os.getcwd()+'/prod-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_prod = open(cfg_file_path)
else:
    sys.exit("prod-cloud-cfg.txt is not in current working directory")

cfg_file_path =  os.getcwd()+'/dev-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_dev = open(cfg_file_path)
else:
    sys.exit("dev-cloud-cfg.txt is not in current working directory")    

secgroups = ['default']

print ("Creating instances ... ")
instance_prod = nova.servers.create(name="prod_server_"+str(identifier), image=image, flavor=prod_flavor, key_name='group_6',userdata=userdata_prod, nics=nics,security_groups=secgroups)
instance_dev = nova.servers.create(name="dev_server_"+str(identifier), image=image, flavor=dev_flavor, key_name='group_6',userdata=userdata_dev, nics=nics,security_groups=secgroups)
inst_status_prod = instance_prod.status
inst_status_dev = instance_dev.status

print ("waiting for 10 seconds.. ")
time.sleep(10)

while inst_status_prod == 'BUILD' or inst_status_dev == 'BUILD':
    print ("Instance: "+instance_prod.name+" is in "+inst_status_prod+" state, sleeping for 5 seconds more...")
    print ("Instance: "+instance_dev.name+" is in "+inst_status_dev+" state, sleeping for 5 seconds more...")
    time.sleep(5)
    instance_prod = nova.servers.get(instance_prod.id)
    inst_status_prod = instance_prod.status
    instance_dev = nova.servers.get(instance_dev.id)
    inst_status_dev = instance_dev.status

ip_address_prod = None
for network in instance_prod.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_prod = network
        break
if ip_address_prod is None:
    raise RuntimeError('No IP address assigned!')

ip_address_dev = None
for network in instance_dev.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_dev = network
        break
if ip_address_dev is None:
    raise RuntimeError('No IP address assigned!')

print ("Instance: "+ instance_prod.name +" is in " + inst_status_prod + " state" + " ip address: "+ ip_address_prod)
print ("Instance: "+ instance_dev.name +" is in " + inst_status_dev + " state" + " ip address: "+ ip_address_dev)

# Host file template
host_conf = "Host dev\nKexAlgorithms +diffie-hellman-group1-sha1\n\tHostName " + ip_address_dev + "\n"
host_conf = host_conf + "\tIdentityFile /home/ubuntu/cluster-keys/RSA-key\n\tUser appuser\n\n"
host_conf = host_conf + "Host prod\n\tKexAlgorithms +diffie-hellman-group1-sha1\n\tHostName " + ip_address_prod + "\n"
host_conf = host_conf + "\tIdentityFile /home/ubuntu/cluster-keys/RSA-key\n\tUser appuser\n"
f = open("/home/ubuntu/.ssh/config", "w")
f.write(host_conf)
f.close()

# Setup ansible hosts file
ansible_hosts = "[servers]\nprodserver ansible_host=" + ip_address_prod + "\n"
ansible_hosts = ansible_hosts + "devserver ansible_host=" + ip_address_dev + "\n\n"
ansible_hosts = ansible_hosts + "[all:vars]\nansible_python_interpreter=/usr/bin/python3\n\n"
ansible_hosts = ansible_hosts + "[prod_server]\nprodserver ansible_connection=ssh ansible_user=appuser\n\n"
ansible_hosts = ansible_hosts + "[dev_server]\ndevserver ansible_connection=ssh ansible_user=appuser\n"
f = open("/etc/ansible/hosts", "w")
f.write(ansible_hosts)
f.close()
