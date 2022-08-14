"""
Problem description (Encrypted, use encryption_helper.py to decrypt):
DncVPlyZXtZAcY07W+Q2y4gEBw9IlKJNtgkCrdA0MJFQ8pNBHI2Mb5x4HLDlwt6bM4o/9DsaU8PT2a2apzaw7g86EXPjFVnFb48koXpEHwtdoFDakMN2838c
newEnQ5L5Eu2YnOSp1cQFHfxy0Fu/Uy+n3puPxOA230/3il4WQB8Fx1ZE+vSxQ0d1AKbs+jjqe4aMCq/JP+CAglGSB57TpNT/Hcr3N6rbcj+e/u5UXAfUUBF
aKV3FhT3DbUDM19nVC9B6WEyp3EKDptCQZreYQpLBJvyAd3hQxUcq7FGT6HniL8pV39+rflk24oWEqMjug6BKMI9LS0ZoxCwWA9/peYA3GrkzISTFb3h/UvR
yPHlNOKwAYQ9ukBkEZtCV7rVNCiqzQbxUfopYQSdUC7+mlJVkvDwK+GSZ38ZqQPTUztKTsgl7KsGJxrjWKT7E0UBTtuUnWzmQlnFqjyvCNZqMSw7M9vJ0YAE
L72WUmWcN0esnpG3J5MYcUf4Qmq5d0rYlEPWCg==
Solution:
12
"""

import optuna
import numpy as np

def objective(trial):
    vote_vec = np.zeros(12, dtype=np.float32)
    vote_vec[0] = 25.
    for i in range(1, 12):
        vote_vec[i] = trial.suggest_float(f"p{i}", 0, 100)
    vote_vec = normalize_vote_vec(vote_vec)
    seats_vec = calc_seats_from_votes(vote_vec)

    return seats_vec[0]

def normalize_vote_vec(vote_vec):
    vote_vec[1:] = 75 * vote_vec[1:] / vote_vec[1:].sum()
    return vote_vec

def calc_seats_from_votes(vote_vec):
    n_parties = len(vote_vec)
    vote_vec[vote_vec <= 5] = 0
    seat_vec = np.zeros(n_parties)
    for i in range(n_parties):
        quotient_vec = vote_vec / (seat_vec + 1)
        seat_vec[np.argmax(quotient_vec)] += 1
    return seat_vec

if __name__ == "__main__":
    timeout = 3.0
    print(f"Running seat optimization for {timeout} seconds...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=timeout)

    n_seats = study.best_value
    vote_vec = study.best_params

    vote_vec_full = np.zeros(12)
    vote_vec_full[0] = 0.25
    for i in range(1, 12):
        vote_vec_full[i] = vote_vec[f"p{i}"]
    vote_vec_full = normalize_vote_vec(vote_vec_full)

    print(f"Best solution is N_seats: {n_seats}, with vote vec: {vote_vec_full}")


