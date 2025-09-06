# POLICY EVALUATION

## AIM

To develop a Python program to evaluate the given policy

## PROBLEM STATEMENT
To find best policy from two policies which are defined by user using policy evaluation function. Where the mdp includes 16 states from 0-15, 0 is the starting state, assigning some 4 random state as holes and 15 is the goal state and then we need to calculate optimal state value function for each state such that we can reach goal using optimal policy using policy evaluation.

## POLICY EVALUATION FUNCTION
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

## OUTPUT:
## polices:
<img width="1193" height="652" alt="Screenshot 2025-09-06 113553" src="https://github.com/user-attachments/assets/88fa6619-d3ad-41de-b33b-4db64ce6744c" />
<img width="1185" height="714" alt="Screenshot 2025-09-06 113619" src="https://github.com/user-attachments/assets/234866fd-709a-46fa-b985-ea4942bb6a81" />
<img width="1193" height="190" alt="Screenshot 2025-09-06 113714" src="https://github.com/user-attachments/assets/90cc890b-ec14-46d8-9de4-12a5717cab73" />

## STATE VALUE FUNCTION:
<img width="1138" height="252" alt="Screenshot 2025-09-06 114317" src="https://github.com/user-attachments/assets/cf4a2bbc-f679-47a8-967c-859572706f16" />
<img width="1149" height="288" alt="Screenshot 2025-09-06 114326" src="https://github.com/user-attachments/assets/99887a25-f37d-4084-aae9-d56ce06841bd" />
## BEST POLICY:
<img width="1114" height="262" alt="Screenshot 2025-09-06 114629" src="https://github.com/user-attachments/assets/4bff66d7-ac9f-49b3-825e-e01fe966c24d" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
