## **Overview**

The meeting focused on refining a GPU scheduling research project away from thermal control and toward work stealing, utilization, and energy proportionality. The discussion clarified that the strongest likely contribution is to **evaluate whether work stealing improves GPU inference** performance, while also using GPU utilization as a proxy for power consumption and operating region.

## **Key Direction for the Project**

The initial idea of scheduling based on GPU physical locality and avoiding overheated neighboring GPUs was deemed impractical because the available student clusters do not expose physical placement information. Instead, the project should focus on metrics that can be observed directly on the GPU, especially utilization and possibly power, since temperature is already tightly regulated by the hardware and cluster software.

A more promising direction proposed was to **build or adapt a GPU work-stealing framework**. In this framing, work stealing would be used primarily for **load balancing**, but could also help the system stay in an energy-proportional operating region by moving tasks off overloaded GPUs and toward GPUs with lighter queues.

## **Why Work Stealing Makes Sense**

The discussion emphasized that GPU work stealing is interesting because it is not widely explored in the same way it has been for CPUs. A GPU version is especially challenging when tasks are tightly synchronized, as the overhead of coordination can outweigh the gains from moving work between devices.

For the current project, **inference** was identified as a good starting point because it is **latency critical** and should benefit from better balancing across GPUs. Training could be used as an additional comparison if time permits, since training is less latency sensitive and may show different behavior because throughput matters more than latency.

## **Metrics to Measure**

For inference workloads, the suggested evaluation metrics were:

* **Latency**  
* Throughput, such as tasks completed per unit time  
* **GPU utilization** on each GPU  
* **Variance or imbalance** in utilization across GPUs in the cluster

The main comparison should be between a **baseline without work stealing** and the **proposed work-stealing approach**. The expectation is that work stealing should reduce queue buildup on overloaded GPUs, **improve latency predictability**, and increase throughput by keeping the cluster in a more efficient operating region.

## **How Utilization and Energy Proportionality Fit In**

**GPU utilization should be treated as an input signal to the scheduler**, not necessarily as the primary objective by itself. The idea is to keep GPUs in a region where utilization is high enough to be **energy proportional**, but not so high that inference latency degrades sharply.

A practical approach suggested in the meeting was to experimentally determine the “**knee**” of the curve by increasing inference load on a single GPU and measuring where **latency begins to rise** steeply. The target operating region was described as roughly above 50–60% utilization, with around 70% often being a reasonable starting point, while 85–90% was described as likely too high for latency-sensitive inference.

## **Recommended Experimental Setup**

The meeting noted that the **Engage** cluster is likely sufficient for development and debugging, even if it only provides access to two GPUs per job. That setup should be enough to validate the logic of the work-stealing framework and catch bugs.

For final evaluation, an eight-GPU setup would be preferable. Since broader access to large shared clusters appears limited unless one is working within a specific lab group, the recommendation was to proceed with the resources currently available and use the larger eight-GPU environment for final results if possible.

## **What to Present and How to Frame the Research**

The project should be presented as a scheduling and performance optimization effort rather than a thermal-management project. The main research question is **whether GPU work stealing can improve inference performance on a cluster of GPUs**, with utilization and energy proportionality serving as important contextual metrics and scheduler inputs.

The suggested presentation structure was similar to a paper talk:

* Problem statement  
* Main contribution  
* Motivation  
* Related work on work stealing and GPU scheduling  
* Design of the framework  
* Evaluation and results

A final presentation length of about 20 to 25 minutes was suggested.

