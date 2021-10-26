###############################################################################
# Pessismistic Open Field RL algorithm
# 
# From Figure 1, Zorowitz et al. (2020)
# Translated from Python to R
# See https://github.com/ndawlab/seqanx for the original code
# 
# Value iteration 
# 
###############################################################################

library(pheatmap)
library(proxy)
library(dplyr)
library(RColorBrewer)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Variables
reward = 10
punishment = -10
epsilon = 0
start = 116

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Environment setup and visualization.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Initialize environment:
size <- 11
grid <- matrix(0, nrow = size, ncol = size)
grid[2,3] <- reward # number 14
grid[2,9] <- punishment # number 20

# Visualize environment:
pheatmap::pheatmap(grid, legend = FALSE, cluster_col = FALSE, cluster_row = FALSE,
                   display_numbers = TRUE, color = colorRampPalette(brewer.pal(n = 7, name = "RdYlBu"))(100))

# Identify coordinates of viable states:
viable_coordinates = data.frame(x = rep(1:size, each = size), y = rep(1:size))

# Make transition matrix with NA's and 1's:
transition_matrix <- proxy::dist(viable_coordinates, viable_coordinates, method = "Euclidean")
transition_matrix <- ifelse(transition_matrix == 1, 1, NA)

transition_matrix[14,] <- NA
transition_matrix[20,] <- NA
transition_matrix[14,14] <- 1
transition_matrix[20,20] <- 1

# Define rewards:
R <- matrix(0, nrow = size*size, ncol = size*size)
R[,14] <- reward; R[14,14] <- 0
R[,20] <- punishment; R[20,20] <- 0
R <- ifelse(transition_matrix == 1, R, NA)

# Define state information:
states <- 1:(size*size)
n_states <- length(states)
viable_states <- states[-c(14,20)] # viable states are states minus terminal states
n_viable_states <- length(viable_states)

# Iteratively define MDP information:
info <- data.frame(S = 0, S_prime = 0, R = 0)
index = 1
for (s in 1:n_states) {

  # What are the possible next states?
  s_prime = which(transition_matrix[s,] %in% 1)  # check on which position are the ones

  # What are the rewards for these next states?
  r = R[s, s_prime]

  # What is the probability of transitioning to next state? 
  # t = 1 # determinisitic 

  for (i in 1:length(s_prime)) {
    info[index,1] = s; info[index,2] <- s_prime[i]; info[index,3] <- r[i]
    index = index + 1
  }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Value iteration and visualization.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Function: Pessimism
pessimism <- function(arr, w) { 
  return(w * max(arr) + (1 - w) * min(arr))
}

# Simulation parameters:
weights <- c(1, 0.5, 0)
gamma = 0.95
tol = 0.0001
max_iter = 100

# Initialize value iteration:
for (weight in weights) {
  
  # Initialize Q-values:
  Q <- matrix(rep(0, nrow(info)), ncol = 1) # Q is the expected reward for every possible transition
  copy <- info 
  
  # Solve for Q-values:
  for (i in 1:max_iter) {
    
    # Make copy:
    q <- Q
    copy$Q <- q
    
    # Precompute successor value: 
    V_prime <- c()
    for (s in 1:(size*size)) {
      arr <- copy %>% filter(S == s)
      V_prime[s] <- pessimism(arr$Q, weight) # V_prime is the value function for every possible state given the policy 
    }
    
    # Compute Q-values:
    for (i in 1:nrow(info)) {
      Q[i] <- info$R[i] + gamma * V_prime[info$S_prime[i]]
    }
    
    # Compute delta:
    delta <- abs(Q - q) 
    
    # Check for termination (i.e., all values of delta below threshold):
    if (all((delta < tol) == TRUE)) { break } 
  }
  
  # Solve for values:
  copy = info
  copy$Q = Q
  
  # Identify max by state:
  V <- c()
  for (s in 1:(size*size)) {
    arr <- copy %>% filter(S == s)
    V[s] <- max(arr$Q) # V is the max value function for every possible state given the policy 
  }
  
  # Fill in terminal states:
  grid_pessism <- matrix(V, nrow = size, ncol = size, byrow = TRUE)
  grid_pessism[2,3] = reward
  grid_pessism[2,9] = punishment
  
  # Visualize environment:
  pheatmap::pheatmap(grid_pessism, legend = FALSE, cluster_col = FALSE, cluster_row = FALSE,
                     display_numbers = TRUE, color = colorRampPalette(brewer.pal(n = 7, name = "RdYlBu"))(100), 
                     main = paste0("Weight: ", weight)) 
  
  # Compute policy: 
  policy <- c()
  policy[1] <- s <- start
  i = 1
  while ( s != 14 && s != 20) {
    
    # Compute optimal q(s, a):
    temp <- copy %>% filter(S == s)
    s <- temp$S_prime[which.max(temp$Q)]
    
    # Append it to policy
    policy <- c(policy, s)
    
    # Make sure it doesn't get stuck: 
    i = i + 1
    if (i > 100) { break }
  }
}

