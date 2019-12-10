import numpy as np
import logging

class AuctionCreator:
    def __init__(self):
        """
            Variables that need values from the user
        """
        self.K = 0
        self.N = 0
        self.R = 0
        self.Smax = 0
        self.epsilon = 0.0
        self.auction_type = ['0: Pure', '1: Levelled Commitment']
        
    
    def get_user_input(self):
        """
            Code to analyze attributes defined in the init section and get specific input from the user
            depending on their types
        """
        for attribute in dir(self):
            if type(getattr(self, attribute)) == int:
                while True:
                    input_value = input("Enter the value for {}: ".format(attribute))
                    try: 
                        input_value = int(input_value)
                        if attribute == "N":
                            if input_value <= getattr(self, 'K'):
                                raise ValueError('Buyer, N should be larger than sellers, K')
                        break
                    except ValueError as error:
                        logging.error(error)
                        logging.error("Invalid Input. Please enter an valid integer value.")
                setattr(self, attribute, input_value)


            elif type(getattr(self, attribute)) == float:
                while True:
                    input_value = input("Enter the value for {}: ".format(attribute))
                    try: 
                        input_value = float(input_value)
                        if input_value < 0.0 or input_value > 1.0:
                            raise ValueError('Epsilon value should be between 0 and 1')
                        break
                    except ValueError as error:
                        logging.error(error)
                        logging.error("Invalid Input. Please enter a valid float value within 0.0 - 1.0")
                setattr(self, attribute, input_value)


            elif type(getattr(self, attribute)) == list:
                while True:
                    input_value = input("Enter the value for {} by selecting the option index  \n {} :".format(attribute, getattr(self, attribute)))
                    try: 
                        input_value = int(input_value)
                        if input_value not in [0, 1]:
                            raise ValueError('The available choices are either 0 or 1')
                        break
                    except ValueError as error:
                        logging.error(error)
                        logging.error("Invalid Input. Please enter a valid option value.")
                setattr(self, attribute, input_value)

    def create_bid_matrices(self):
        """
            Create and Initialize Matrices based on the user specified dimensions.
            Seller price matrix K x R
            Alpha Values N x K
            Buyer price matrix N x K x R
        """
        self.seller_price = np.random.rand(self.K, self.R) * self.Smax
        self.alpha_factors = np.random.rand(self.N, self.K) + 1 #Greater than 1
        self.buyer_price = np.zeros((self.N, self.K, self.R))
        
        number_of_rounds = self.seller_price.shape[1]
        for round in range(number_of_rounds):
            self.buyer_price[:, :, round] = self.alpha_factors * self.seller_price[:, round]


    
        return self.seller_price, self.buyer_price
    


if __name__ == "__main__":
    auction_input = AuctionCreator()
    auction_input.get_user_input()  
    auction_input.create_bid_matrices()          
