# gym-sip
    Credits:
        Gym environment adapted from https://github.com/bioothod/gym-stocks
        Deep-Q network adapted from https://github.com/MorvanZhou/pytorch-A3C
        Thank you to bioothod who enabled me to easily get started using gym.
        Thank you to MorvanZhou for his incredible PyTorch examples.


# Quick start:
    
    gym-sip is a custom gym environment that is configured to take in NBA money lines. 
    Actions are decided based on learned previous examples, stored in memory of N states.
    
    Once you have git cloned the project, create a folder in gym-sip called 'data' and place
    a csv in the folder. This folder must meet certain specifications or requires modification 
    to the source code. Mainly the columns must be correct.
    
    Run test.py
    
    
    
    
    
# CSV column specifications
    game_id               
    cur_time             
    secs                  
    a_pts                 
    h_pts
    status
    a_win
    h_win
    last_mod_to_start
    num_markets
    a_odds_ml
    h_odds_ml
    a_hcap_tot
    h_hcap_tot
