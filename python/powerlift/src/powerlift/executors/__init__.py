""" Executors that run trials in their environment.
"""

from powerlift.executors.azure_ci import AzureContainerInstance
from powerlift.executors.docker import InsecureDocker
from powerlift.executors.localmachine import LocalMachine
