import os 

def a2p_mkdir(home_dir):

    '''
    Inputs:
        Home directory (home_dir) is in the form of .../YYYY-MM-DD_MouseID_DXSX
    '''

    # Change to the home directory
    os.chdir(home_dir)

    sub_dirs = [
            "01_Raw_Data",
                "01_Raw_Data/imaging",
                "01_Raw_Data/metadata",
            "02_S2P_Processed_Data",
        ]
        
    # Creating main directories
    for i in sub_dirs:
        os.makedirs(i, exist_ok=True)

if __name__ == "__main__":
    home_dir = input("Please enter the home directory (e.g. .../YYYY-MM-DD_MouseID_DXSX): \n")
    a2p_mkdir(home_dir)
    print("Directories created successfully.")
        
        