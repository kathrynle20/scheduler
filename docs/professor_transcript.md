We considered the utilization of the GPU and the temperature of the GPU and then the one thing a teammate suggested was actually to think about the ocality of the GPU as in the heat from say a group of GPUs in the same kind of immediate like group like in a physical sense can amplify the issue of like having overheating and they're thinking of actually scheduling based on that as in if you have a gpu that's already kind of overheated or generating a lot of heat in a specific group then you should not schedule more tasks on gpus in that group but at least from what I understand of the clusters available to us students, we can't really decide or we don't have access to the physical location of the GPUs in a cluster.  
Speaker 2

What machines are you using for the experiments?  
Speaker 1

We were just using Engage, engaging right now, currently. And I guess one of our teammates is in a CTO lab that has access to their own cluster. So just those two.  
Speaker 2

Okay. Right. So you won't be able to know the physical placement. You might be or should be able to read the temperature the GPU is operating at.  
Speaker 1

But  
Speaker 2

what you're trying to do, so you would be able to... like decide depending on the temperature what type of jobs to place on that gpu but not what the neighboring gpus are doing yes correct so if what you want to do is like just single gpu or make the decisions at a single gpu granularity then that's fine um i think what i had said when we also met for the project the first time is that these systems have fairly good thermal regulation  
mechanisms so it wouldn't actually let you go over the thermal limit or even very close to the thermal limit. One thing you can do is so temperature is usually a good proxy for power and power is a good proxy for utilization for the GPUs not for everything. So one thing you can do is if you want to do is a distributed setup where you're trying to maybe load balance or you're trying to operate in a specific power region so that you get a more energy proportional part. of the GPU's operation you could look at power numbers instead and those are either you can read them directly from the GPU or there are tools that are models where you give them the GPU utilization it will tell you what's the power consumption or you could even use the GPU utilization as a proxy compiler. I think that might be a little bit more interesting and more useful to do than trying to not exceed the thermal limit or trying to not get too close to the thermal limit of the GPU because there's already a lot of tools that do that.  
Speaker 1

Okay. Okay. That makes sense. Yeah. That was actually the main reason I wanted to talk because as we were doing these tests, we were thinking that like a lot of these techniques we were trying to attempt for the scheduling seem to have already been done pretty like well in the industry and research. So yeah. I wanted to ask, do you have any suggestions on interesting things?  
And you said one just now, but anything else that you can think of that's interesting to tackle in the sense of scheduling.  
Speaker 2

Right. So it depends on what application you're running. If what you're running is having a distributed training or inference framework, we can look at work stealing mechanism. **So trying to port that on GPUs, that hasn't been done that much because usually these applications have a lot of synchronization and they're very fine-grained.** So in that case, the work stealing doesn't work very well because you have to **synchronize and make sure that you don't have multiple GPUs working on the same thing**. So you could try evaluating some simple work stealing mechanisms and you can look at CPU-based papers that propose work stealing and try and adapt some of that for a GPU cluster.  
I think that would be useful. And the benefit of that is So what you can do is you can have the utilization region or the power consumption region as an input to the workstealing framework. So you can basically say steal tasks from another GPU until all the GPUs are running at 50 to 70% of the utilization because that is the energy proportional region and I want to keep the cluster at that point. And that would be a nice framework to have and you would also be able to show whether workstealing is effective for GPUs or not.  
Speaker 1

Gotcha. So the idea is to use work stealing to manage the energy proportionality to make sure all the GPUs in a group or cluster is running efficiently and not just a subset of GPUs that are doing all the work.  
Speaker 2

So workstealing is mostly for load balancing, but it would help with any disproportionality as well, because if one GPU is overloaded because it has a very long queue of events, that will push it closer to 100%, and then your performance starts to degrade, and then you get also close to the thermal limits, although you might not be able to control that. So workstealing is more for load balancing, but depending on how you tune the workstealing, so depending on how many tasks you tell it to steal, or what are the... the metrics that it considers in the inputs for example if you're giving the gpu utilization or the power as inputs then it can still until the point where all gpus are load balanced in the region that you want them to run  
Speaker 1

gotcha so i guess in my um head like the energy proportionality and the kind of load balance, they are already not closely correlated, as in if a GPU is being used a lot, then their energy proportionality will be higher intuitively.  
Speaker 2

So there are different concepts. Load balancing means that I want all my GPUs to be at about the same utilization. And then energy proportionality means um i want to run in a region where the power i consume is proportional to the performance i'm getting so i could have a load balance cluster but everything is at 10 okay so that's not a that's not a very emotional point because you're consuming a lot of power for not a lot of performance um i can also have a load balance cluster that is a 70 so that is a good operational point because at that region in that region the gpus and the cpus are more energy proportional you want something like that we don't want a lot of imbalance between the GPUs because then your performance will suffer, some will be overloaded. You also don't want them to run at very low utilization because then you don't get the energy proportionality. So you're trying to balance the two.  
Speaker 1

Okay. So if we were to go into this area of the scheduling problem, what would you say are some important metrics to measure and analyze to show that either this works or this doesn't work?  
Speaker 2

Right. So you want to show... So what is the application you're on?  
Speaker 1

We are right now just running a simple inference task.  
Speaker 2

Okay. So you would want to show latency, so day latency, for the inference tasks. You can show how many tasks per cycle you are running. And you can show what is the GPU utilization and then the variance across utilization within the clusters. how balanced is the GPU utilization across the different servers in the cluster. And then you show that without warp stealing and with warp stealing. So ideally with warp stealing, you'll get more predictable performance because you don't have long queues in some of the GPU, so your day latency should go down. And then you should also get higher throughput because you're operating in a region where the system is more energy proportional, so for the same power budget. you should be able to get more performance.  
Speaker 1

Okay, okay, that makes sense.  
Speaker 2

Those are easy metrics to measure. You can measure those from the client. So whatever script you have that is sending the inference cast, you can measure latency and throughput from there. Okay,  
Speaker 1

okay. So if we were to kind of shift gears to the work stealing idea, then do we even need to like... worry about a scheduler that is based on this peer utilization? Is that relevant to this work stealing case? Or should we just only focus on no work stealing metrics and work stealing metrics?  
Speaker 2

So your work stealing scheduler would need to be aware of the utilization because it needs to run all the GPUs within a certain region. So somewhere else, you know. 50 to 70, 80% is where you want to be. So you need to at least measure it so that you can tell the work stealing framework how many tasks to steal and from where based on that.  
Speaker 1

Okay. Okay. That makes sense. This is me thinking like... off the top of my head but like why would works I don't understand why work stealing would be bad in like any case as in like if there was a naive scheduler that did not utilize like certain gpus and were concentrating all the work on one region of gpus then would it automatically intuitively be better if that work was more distributed across the entire cluster or there are other factors I'm not considering here  
Speaker 2

So usually work stealing improves things a lot. Where it doesn't improve is if you have a job where you have a lot of synchronization between tasks. So let's say I have two GPUs, right? And one has a queue of 10 tasks and the other has a queue of four tasks. So ideally I could take three tasks from GPU 1 and put them on GPU 2\. And then I have a model of balance, assuming all tasks are about the same length. Now, if every time I have to move a task from one GPU to the other, I have to synchronize who is running what, then the overhead of synchronization becomes bigger than the benefit you get from workstation. So that happens sometimes with ML tasks because they have very fine-grained synchronization. Depending on what framework you're using, they might or they might not. I guess for inference, they shouldn't. This would be more for training. So we could see benefits. If you have time, you could compare a training job to an inference job and then see which one it benefits more.  
Speaker 1

Okay. Okay. Like, I guess, like the general idea is just to see for... a task that we choose if work ceiling works in GPUs. But if we have time, we can then add another dimension of does work ceiling do good in tasks that have synchronization requirements. Okay. Okay. Yep. That doesn't make sense. And then another question is about the testing. So with our teammate that has access to her labs clusters, they it's easy for her to access i think eight gpus at a time for a job but at least from what i've tested on engaging the most i can get access to for a single job is two gpus and i assume that we should probably have at least like four or more GPUs to actually see these results in like work stealing or any type of scheduling. Do you have any recommendations on where we can get access to clusters or jobs that allow us access to more GPUs?  
Speaker 2

I think the engaged cluster is the main thing for MIT. I mean, individual groups have GPUs, but I don't think you'd be able to get access to those. are in that group. I think that's fine for testing, like for the development phase of the work-stealing framework, you can do that with two GPUs. I don't have the bugs. And then for the final results, you run that on eight GPUs.  
Speaker 1

Oh, okay. Okay. Yeah. That makes sense. You are in CSAIL, correct? Yes. So I am a grad student in CSAIL. Do you know if there is like separate clusters that kind of available among the CSAIL like kind of graduate community?  
Speaker 2

per professor,  
Speaker 1

like there aren't  
Speaker 2

GPU clusters. But unless you're working with a group, they usually don't make them publicly or at all.  
Speaker 1

Okay, that makes sense. Yeah, I don't think, unfortunately, I don't think my group has access to clusters.  
Speaker 2

Yeah, that's fine. I mean, you can do for development, you can use two GPUs. That will help you find any bugs. And then you run the eight is enough. With eight, you should be able to see whether it works in-house or not. Okay, okay.  
Speaker 1

And to summarize our conversation, what we are doing, instead of focusing on thermals, because like you said, thermals are pretty restricted by the GPUs or the systems themselves, so we can't really do much minute control in that situation. So we should switch over to more of a... focus on energy proportionality and utilization,  
which is kind of a proxy for the GPUs. Sorry, you said it was a proxy for the GPUs, which metric again?  
Speaker 2

So if you look at utilization for GPUs because of the type of tasks they run, it's a good proxy for power consumption. That's not the case for everything, but for GPUs, if you're running very close to 100%, That's a good proxy for 100% power consumption as well.  
Speaker 1

Gotcha. Okay. Gotcha. Yeah. And then, so those are the kind of metrics we're trying to measure to inform a work-stealing algorithm, which is a type of scheduling, to see if there's benefits with that type of scheduling. But like kind of the fundamental idea of the work-stealing algorithm is to improve utilization and like... I guess, energy professionality across the entire culture?  
Speaker 2

Workstealing is mostly performance optimization because you take from GPUs that have long queues and you put tasks in GPUs that have short queues, you have less load imbalance. So that should help performance. But then if in addition to that, you tune the workstealing... so that the utilization of the GPUs are in the energy proportional region, which is above 50%, but not quite a hundred percent, then that will also help with power consumption.  
Speaker 1

Okay. Okay. So let me make sure I understand the kind of goals or metrics we're trying to maximize or improve on is energy proportionality, which is kind of like correlated with your power consumption. And also performance?  
Speaker 2

Performance, yes. So for performance, if you're doing inference, that will be the latency. And if you're doing training, it will be throughput. Okay.  
Speaker 1

Gotcha. And the utilization here isn't really a metric we're trying to maximize because like 100% utilization. but might mean like worse energy proportionality that's just a metric that is used in the work stealing algorithm to help max or i guess improve the other two metrics we just mentioned right  
Speaker 2

so you use it as an input to your work stealing um 100 utilization is actually not bad for energy proportionality but it's bad for performance because this is these are latency critical tasks if you run 100 They're going to get badly. For proportionality, it's fine. But you want to balance between proportionality and performance. Okay.  
Speaker 1

This might be a subjective question. So right now, we've only used inference tasks for our testing. But do you have any thoughts on if that is the right choice to test if a work stealing algorithm is usable or a good fit for GPUs or not?  
Speaker 2

That's good, because those are the ones that are more latency critical. If you have time, it would be nice to compare between inference and training, because for training you care about throughput, so it's not as latency critical as inference, and usually distributing the load over more GPUs helps, while in the case of inference you are limited by the performance of a single GPU.  
So it would be nice to compare the two if you have time, but focusing more on inference is good. Okay,  
Speaker 1

that makes sense. And remind me again for the utilization as an input, is there a utilization that we should aim to achieve per GPU? Or again, does that number change depending on the kind of workload environment?  
Speaker 2

It will depend. So you have to look at, as you push utilization higher, how does the latency of your inference tasks increase? If, let's say, going over 70% means that your latency increases a lot, then you should copy that 70%. Usually 70% should be fine. 70% should be fine. If you go more than 85, 90, then your latency is going to increase a lot. But it will depend on the specific task. So what you can do is... send more inference tasks per second.  
So have a script that sends more inference tasks. You can do it on a single GPU. So you send more inference tasks to the GPU and then you see what happens to the latest. So let's say I send 10 inference tasks per second. My latency is good. I sent 15\. My latency is still good. I sent 500 and now my latency has increased a lot. So you want something that it looks like a hockey stick. So the latency is low as you increase the load and then at some point it increases exponentially. You want to operate somewhere before the exponential because the moment latency increases that much, the system is unstable.  
Speaker 1

Okay, that makes sense. So Another point I kind of was confused about was if we are using utilization as kind of a proxy for power consumption, but we're trying to maximize power or energy proportionality, then how are we supposed to measure if our algorithm generates good energy proportionality or not?  
Speaker 2

So energy proportionality is not It's not a specific metric. Energy proportionality means that you are running in a region where the power increases proportionally to your load, to your utilization. So you can measure, I don't know exactly if this GPU will let you measure power directly, but you can definitely measure GPU utilization. So as long as you are over... 60 50 60 percent in the gpu utilization it will be energy proportional and then the question is i don't want to get so high in utilization that my performance will suffer so you have to find what is the right trade-off you know so you are within the region where it is energy proportional but it's not so high that your performance degrades yes  
Speaker 1

okay that all makes sense so to kind of for the very simple terms the main goal of this research and testing is to see if work stealing can improve performance and then like our inputs like utilization and like considerations like energy proportionality those are just like metrics that to consider as like either inputs or outputs to this algorithm but the fundamental idea is can we um use a work stealing algorithm to show that we can improve performance of uh inference tasks on a cluster of gpus right  
Speaker 2

okay  
Speaker 1

okay yep that makes sense but yeah i think that makes sense and that gives me and probably my team better direction on what to do um yeah i think that covers all my questions all  
Speaker 2

right sounds good awesome  
Speaker 1

and i guess the last sorry um maybe you haven't thought this far but do you have an idea of how long each of the presentations will be or how long they need to be  
Speaker 2

about 20 minutes i think 20 to 25 minutes okay  
Speaker 1

i don't know if this is on the website but is there like kind of a guideline of what you expect to be put in the slides yeah  
Speaker 2

so it will be similar to the paper presentation so you want like one slide that says What is the problem that you're trying to solve? What is the main contribution you've made? Like a summary of the project. And then you can go through motivation and why the problem is important, related work, what people have done before. So in your case, you can describe work stealing frameworks or other GPU scheduling frameworks. Then you talk about the design of your framework and then the evaluation.  
Speaker 1

Okay, that makes sense. Okay, cool. Thank you so much. I appreciate it.  
Speaker 2

All right. Thank you. Bye. Bye. Have a good day.

