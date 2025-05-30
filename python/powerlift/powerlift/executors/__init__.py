"""Executors that run trials in their environment."""

from powerlift.executors.bicep_azure_ci import BicepAzureContainerInstance
from powerlift.executors.azure_ci import AzureContainerInstance
from powerlift.executors.azure_vm import AzureVMInstance
from powerlift.executors.docker import InsecureDocker
from powerlift.executors.localmachine import LocalMachine
