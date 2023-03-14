def test_log_file(log_file,name,test_val,n_iter):
    """
    Args:
        log_file (str): Path to the log file
        name (str): Name of the value to test
        test_val (float): Value to test against
        n_iter (int): Number of iterations to test over
    
    Returns:
        bool: True if all of the values are greater than test_val, False otherwise
    """
    # Read the log file
    with open(log_file, "r") as f:
        lines = f.readlines()

    # Extract the values of name for the last n iterations
    scalar_list = []
    for line in reversed(lines):
        if name in line:
            values = line.split(": ")
            scalar = float(values[2])
            scalar_list.append(scalar)
            if len(scalar_list) == n_iter:
                break
    
    # Check if all of the values are greater than test
    return all(return_val > test_val for return_val in scalar_list)


def test_log_file_custom_env(log_file):
    """
    Args:
        log_file (str): Path to the log file
    
    Returns:
        bool: True if bc_return is 30% lower than expert_return, False otherwise
    """
    # Read the log file
    with open(log_file, "r") as f:
        lines = f.readlines()

    # Extract the values of name for the last n iterations and check if it is lower than 30% of the expert policy
    expert_return = 0
    bc_return = 0
    for line in reversed(lines):
        if "Initial_DataCollection_AverageReturn" in line:
                values = line.split(": ")
                scalar = float(values[2])
                expert_return = scalar
        if "Eval_AverageReturn" in line:
                values = line.split(": ")
                scalar = float(values[2])
                bc_return = scalar
                
    # Check if bc_return is lower than 30% of the expert return
    if bc_return < (expert_return * 0.3):
        return True
    else:
        return False


if __name__ == "__main__":
    # Question 1.2 Ant behavior cloning
    assert test_log_file("ant1-2.log", "Eval_AverageReturn", 1500, 1) == True, "The average return is not greater than 30% of the expert policy." 
    # Question 1.2 Custom env
    assert test_log_file_custom_env("custom1-2.log") == True, "The average return is not smaller than 30% of the expert policy."
    # Question 2.2 DAgger 
    assert test_log_file("dagger_ant2-2.log", "Eval_AverageReturn", 4000, 5) == True, "The average return is not great enought, you should at least reach 4000 for DAgger."
    
    
    