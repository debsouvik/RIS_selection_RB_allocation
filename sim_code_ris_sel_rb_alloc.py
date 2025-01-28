def recsum(array):
  sum=0.0
  for x in array:
    sum+=1/(x+1e-10)
  return sum

import numpy as np
from scipy import stats
import math

class UCB_MAB_Agent:
    def __init__(self, n_arms):
        """
        Initialize the UCB agent.

        Parameters:
        n_arms (int): Number of arms (actions).
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm has been pulled
        self.values = np.zeros(n_arms)  # Estimated value of each arm

    def select_arm(self,n_agents):
        """
        Select an arm to pull using the UCB algorithm.

        Returns:
        int: The index of the selected arm.
        """
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            # Pull each arm once before applying UCB formula
            return int(total_counts)

        ucb_values = self.values + np.sqrt( np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """
        Update the estimated value of the chosen arm.

        Parameters:
        arm (int): The index of the chosen arm.
        reward (float): The observed reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

    def return_max(self):
      return np.max(self.values)
    
def sic_decoder(channel_coefficients, power_allocations, noise_power):
    """
    Perform Successive Interference Cancellation (SIC) for N users in a downlink scenario.

    Parameters:
    channel_coefficients (list or np.array): Channel coefficients for N users (increasing order).
    received_signal (float): The received signal at the base station.
    power_allocations (list or np.array): Power allocated to each user (decreasing order).
    noise_power (float): The power of the noise in the system.

    Returns:
    list: SINR (Signal-to-Interference-plus-Noise Ratio) for each user.
    """
    n_users = len(channel_coefficients)
    sinr_values = []


    for i in range(n_users):
        # Signal power of the current user
        signal_power = channel_coefficients[i] ** 2 * power_allocations[i]

        # Interference power from remaining users
        interference_power = np.sum([channel_coefficients[j] ** 2 * power_allocations[j] for j in range(i + 1, n_users)])

        # Calculate SINR for the current user
        sinr = signal_power / (interference_power + noise_power)
        sinr_values.append(sinr)


    return sinr_values

import random
import numpy as np

def generate_unique_points(n, x_range=(-100, 100), y_range=(-100, 100)):
    """
    Generate n unique points in 2D space such that no points overlap.

    Parameters:
    n (int): Number of points to generate.
    x_range (tuple): Range of x-coordinates as (min, max).
    y_range (tuple): Range of y-coordinates as (min, max).

    Returns:
    list: A list of tuples representing the points.
    """
    points = set()

    while len(points) < n:
        x = random.randint(*x_range)
        y = random.randint(*y_range)
        if (x,y)!=(0,0):
         points.add((x, y))  # Add the point to the set to ensure uniqueness

    return list(points)

def generate_binary_list(probabilities):
    """
    Generate a list of 0s and 1s based on input probabilities.

    Parameters:
    probabilities (list or np.array): An array of probabilities (values between 0 and 1).

    Returns:
    list: A list of 0s and 1s, where each element is sampled based on the corresponding probability.
    """
    return [1 if np.random.rand() < p else 0 for p in probabilities]

import numpy as np

def calculate_channel_gain(user_coords, bs_coords, m, pathloss_exponent):
    """
    Calculate the channel gain between a user and a base station (BS).

    Parameters:
        user_coords (tuple): Coordinates of the user as (x, y).
        bs_coords (tuple): Coordinates of the base station as (x, y).
        m (float): Nakagami-m fading parameter (shape factor).
        pathloss_exponent (float): Path loss exponent.

    Returns:
        complex: Channel coefficient (complex value).
    """
    # Step 1: Calculate the Euclidean distance between the user and BS
    distance = np.sqrt((user_coords[0] - bs_coords[0])**2 + (user_coords[1] - bs_coords[1])**2)

    if distance == 0:
        raise ValueError("User and BS cannot have the same coordinates.")

    # Step 2: Apply path loss model (distance^-pathloss_exponent)
    path_loss = distance ** -pathloss_exponent

    # Step 3: Generate Nakagami-m fading
    fading_amplitude = np.random.gamma(m, 1/m)  # Nakagami-m fading amplitude
    phase = np.random.uniform(0, 2 * np.pi)    # Random phase

    # Combine fading amplitude and phase to generate the complex fading coefficient
    fading_coefficient = fading_amplitude * np.exp(1j * phase)

    # Step 4: Calculate the channel coefficient
    channel_coefficient = fading_coefficient * np.sqrt(path_loss)

    return (channel_coefficient)

def dynamic_probabilites(RIS_list,UE):
  mu=2
  dh=1
  hh=2 ## height of obstcale
  hr=10 ## height f RIS
  lamh=0.2
  hu=1.3 ## height of UE
  V=2
  exponent=lamh*dh*((hh-hu)/(hr-hu))
  exponent=exponent*V*2/math.pi
  probabilities=[]
  for ris in RIS_list:
    d= np.sqrt((ris[0] - UE[0])**2 + (ris[1] - UE[1])**2)
    alpha=exponent*d
    p=(alpha)/(alpha+mu)
    probabilities.append(1-p)
    #print(1-p,"distance ",d)
  return probabilities


No_ris=10
No_RB=20
No_UE=40
No_elements=100
No_t_antenna=64
communication_block=50
T_power=39 #Watts
f=28e9
velocity=0.8
wave=3e8/f
Tc=wave/(16*math.pi*velocity)
Bc=10e6#(100e9)
Nc=Tc*Bc
K_factor=10
naka_m=(K_factor+1)*(K_factor+1)/(2*K_factor+1)
thresholds=[4,5,6,7,8,9,10]
band=100 ##Mhz

def csiblock(n_ris,n_ue,n_elements):
  Tp=int((n_ris*n_elements+1)+(n_ris*n_elements/No_t_antenna +1)*(n_ue-1))
  return Tp

import math
from scipy.stats import bernoulli

def find_min_tuple(lst):
  min=10000000
  for tup in lst:
    if tup[0]<min:
      min=tup[0]
  return min


def RB_allocation(sr_matrix,nRB,nUE):
  count=len(sr_matrix)
  sumrate=[0 for i in range(nUE)]



  while(count<nRB):
    sum1=[]
    for s in sr_matrix:
      minimum=find_min_tuple(s)
      sum1.append(minimum)
    user=sum1.index(min(sum1))

    sumt=0
    for i in range(len(sr_matrix[user])):
      sr_matrix[user][i][0]+=sr_matrix[user][i][0]
      sumt+=sr_matrix[user][i][0]
   

    count+=1
  for s in sr_matrix:
    for l in s:
      sumrate[l[1]]=l[0]
  return sumrate #### returns the list of all rates of all UEs


def RB_allocation_comp(sr_matrix,nRB,nUE):
  count=len(sr_matrix)
  sumrate=[0 for i in range(nUE)]
  sum1=[]
  for s in sr_matrix:
    sumt=0
    for tup in s:
      sumt+=tup[0]
    sum1.append(sumt)

  while(count<nRB):
    user=sum1.index(max(sum1))
    sumt=0
    for i in range(len(sr_matrix[user])):
      sr_matrix[user][i][0]+=sr_matrix[user][i][0]
      sumt+=sr_matrix[user][i][0]
    sum1[user]=sumt

    count+=1
  for s in sr_matrix:
    for l in s:
      sumrate[l[1]]=l[0]
  return sumrate

def compute_sum_rate_mab(arm,agents,pm,dc,RIScm):
  los=bernoulli.rvs(0.5, size=1)[0]
  sum_rate=[0 for i in range(len(dc))]
  channel_list=[]
  power_list=[]
  for i in agents:
    ## create channel list and power list
    p_v=bernoulli.rvs(p=pm[i][arm],size=1)[0]
    channel=abs(p_v*RIScm[arm][i]+los*dc[i])
    channel_list.append(channel)
  tot=recsum(channel_list)
  for i in range(len(channel_list)):
    if channel_list[i]==0 or tot==0:
      power=0
    else:
      power=((1/channel_list[i])/tot)*(T_power/No_ris)
    power_list.append(power)
  chl=sorted(channel_list)
  pw=sorted(power_list,reverse=True)
  sinr_list=sic_decoder(chl,pw,1e-17)
  Tc=csiblock(1,No_UE,No_elements)
  for i in range(len(sinr_list)):
    sum_rate[i]=band*(1-(Tc/Nc))*math.log2(1+sinr_list[i])
  return sum_rate



def compute_sum_rate(agents,pm,dc,RIScm):
  los=bernoulli.rvs(0.5, size=1)[0]
  sum_rate=[0 for i in range(len(dc))]
  channel_list=[]
  power_list=[]
  for i in agents:
    channel=los*dc[i]
    for j in range(No_ris):
    ## create channel list and power list
      p_v=bernoulli.rvs(p=pm[i][j],size=1)[0]
      channel+=p_v*RIScm[j][i]
      channel=abs(channel)
    channel_list.append(channel)
  tot=recsum(channel_list)
  for i in range(len(channel_list)):
    if channel_list[i]==0 or tot==0:
      power=0
    else:
      power=((1/channel_list[i])/tot)*(T_power/No_ris)
    power_list.append(power)
  chl=sorted(channel_list)
  pw=sorted(power_list,reverse=True)
  sinr_list=sic_decoder(chl,pw,1e-17)
  Tc=csiblock(No_ris,No_UE,No_elements)
  for i in range(len(sinr_list)):
    if Tc/Nc<1:
      sum_rate[i]=band*(1-(Tc/Nc))*math.log2(1+sinr_list[i])
    else:
      sum_rate[i]=0
  return sum_rate

def confidence_interval(data, confidence=95):
    data_array = np.array(data)
    mean = np.mean(data_array)
    n = len(data_array)
    std_err = stats.sem(data_array)  # Standard error of the mean
    if confidence==95:
      Z=1.960
    elif confidence==99:
      Z=2.576
    else:
      print("wrong confidence level")
      exit()
    margin_of_error = Z * std_err/math.sqrt(n)


    return margin_of_error

def pop_random_elements(original_list, n):
    if n > len(original_list):
        raise ValueError("n cannot be greater than the length of the original list")

    # Randomly pop n elements
    popped_elements = []
    for _ in range(n):
        index = random.randint(0, len(original_list) - 1)
        popped_elements.append(original_list.pop(index))

    return popped_elements

from scipy.stats import bernoulli

agents = [UCB_MAB_Agent(No_ris) for i in range(No_UE)]

n=No_ris+No_UE

L=math.ceil(No_UE/No_RB)
integers=list(range(0,No_UE))
grouping_two=[integers[i:i+L] for i in range(0,len(integers),L)]

points=generate_unique_points(n,(-100,100),(-100,100))
RIS = pop_random_elements(points,No_ris)  # First No_RIS points
UEs = points  # Remaining points
optimal_arm=[]
probability_matrix=[]

direct_channel=[calculate_channel_gain(ue,(0,0),naka_m,4) for ue in UEs]
channel_matrix=[]
for r in RIS:
  a=calculate_channel_gain(r,(0,0),naka_m,2)
  temp=[]
  for ue in UEs:
    b=calculate_channel_gain(ue,r,naka_m,2.5)
    temp.append(a*b)
  channel_matrix.append(temp)

q=0
for ue in UEs:
  temp1=dynamic_probabilites(RIS,ue)
  m=temp1.index(max(temp1))
  optimal_arm.append(temp1[m]*abs(channel_matrix[m][q]))
  q+=1
  for i in range(len(temp1)):
    if i!=m:
      temp1[i]*=random.uniform(0.6,0.9)
  probability_matrix.append(temp1)

optimal_chosen=[0 for _ in range(len(UEs))]
for i in range(len(UEs)):
  lst=[]
  for j in range(len(RIS)):
    lst.append(abs(probability_matrix[i][j]*channel_matrix[j][i])**2)
  optimal_chosen[i]=lst.index(max(lst))

selected_arm=[0 for an in agents]
cumulitive_regret=0
average_compare_rate=[]
expected_reward=0

for b in range(1000):

  grouping={i:[] for i in range(No_ris)} ## group of UEs based on RIS

  i=0
  for an in agents: ## all agents choose an RIS and sorted into groups
    arm=an.select_arm(len(UEs))
    selected_arm[i]=arm
    grouping[arm].append(i)
    i+=1
  ## the groups of UEs are now formed. It is time to communicate
  rewards=[0 for i in range(No_UE)]
  optimal_rewards=[0 for i in range(No_UE)]
  sum_rate_mab=[0 for i in range(No_UE)]
  sum_rate_comp=[0 for i in range(No_UE)]
  sum_rate_RB=[0 for i in range(No_UE)]
  sum_rate_greedy=[0 for i in range(No_UE)]
  rballoc_mab=0
  rballoc_comp=0
  for t_iter in range(communication_block):
    groups=[]
    groups_comp=[]
    #p_matrix=binary_generator(probability_matrix)
    ### for each UE group return the group sum and then concatanete the lists and keep adding to the sumrate lists
    #### also generate the reward for each UE (check the channel matrix from the indecies of the grouping disctionary)
    temp3=[]
    temp4=[]

    for key, value in grouping.items():
      if len(value)!=0:
        for v in value:
          p_v=bernoulli.rvs(p=probability_matrix[v][key],size=1)[0]
          rewards[v]+=1e5*T_power*abs(p_v*channel_matrix[key][v])**2
          if key==optimal_chosen[v]:
            op_v=p_v
          else:
            op_v=bernoulli.rvs(p=probability_matrix[v][optimal_chosen[v]],size=1)[0]
          optimal_rewards[v]+=1e5*T_power*abs(op_v*channel_matrix[optimal_chosen[v]][v])**2
        temp3=compute_sum_rate_mab(key,value,probability_matrix,direct_channel,channel_matrix)
        y=[]
        for l,v in zip(temp3,value):
          y.append([l,v])
        groups.append(y)
        #for l,v in zip(temp3,value):
          #sum_rate_mab[v]+=l
    ## was commented out below
    for lst in grouping_two:
      temp4=compute_sum_rate(lst,probability_matrix,direct_channel,channel_matrix)
      z=[]
      """for l,v in zip(temp4,lst):
        z.append([l,v])
      groups_comp.append(z)"""

      for l,v in zip(temp4,lst):
        sum_rate_comp[v]+=l #### comment end

    ### This is where the changes happened######
    """for key, value in grouping.items():
      if len(value)!=0:
        temp4=compute_sum_rate(value,p_matrix,direct_channel,channel_matrix)
        z=[]
        for l,v in zip(temp4,value):
          z.append([l,v])
        groups_comp.append(z)"""

    temp5=RB_allocation(groups,No_RB,No_UE) ### this the allocation for MAB
    for i in range(len(temp5)):
      sum_rate_mab[i]+=temp5[i]

    #temp6=RB_allocation_comp(groups_comp,No_RB,No_UE)
    #for i in range(len(temp6)):
      #sum_rate_comp[i]+=temp6[i]"""
  #sum_value=0
  for j in range(len(agents)):
    agents[j].update(selected_arm[j],(rewards[j]/communication_block))



    #sum_value+=agents[j].values[selected_arm[j]]  agents[0].return_max()
  cumulitive_regret+=((optimal_rewards[0]/communication_block)-(rewards[0]/communication_block))
  expected_reward=(1-1/(b+1))*expected_reward+(1/(b+1))*sum(rewards)/communication_block



  if b%100==0 or b==1000:
    print("wosrt user of MAB and comp ",min(sum_rate_mab),min(sum_rate_comp))
    print("mab rate (",No_UE,",",sum(sum_rate_mab)/1e6,")")
    print("comp rate (",No_UE,",",sum(sum_rate_comp)/1e6,")")
    print("proportion mab ", [sum(1/No_UE for value in sum_rate_mab if value >= threshold) for threshold in thresholds])
    print("proportion_comp ",[sum(1/No_UE for value in sum_rate_comp if value >= threshold) for threshold in thresholds])

