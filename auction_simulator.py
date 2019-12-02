from auction_creator import AuctionCreator
import logging
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set level=None to remove display messages
logging.basicConfig(level=logging.INFO)

class AuctionSimulator:
    def __init__(self, seller_prices, buyer_prices, auction_type):
        """
            Create seller profits/ buyer profits array and define other variables 
        """
        self.auction_type = auction_type
        self.seller_prices = seller_prices
        self.buyer_prices = buyer_prices
        self.seller_profits = np.zeros((self.seller_prices.shape[0]))
        self.buyer_profits = np.zeros((self.buyer_prices.shape[0]))
        self.number_of_rounds = self.buyer_prices.shape[2]
        self.number_of_auctions = self.buyer_prices.shape[1]
        self.auction_results = {}
        self.market_price_developments = []
        


    def run_auctions(self):
        """
            Run pure or levelled commitment auction
        """
        if self.auction_type == 0:
            self.run_pure_auction()
        else:
            # Levelled Commitment auction goes here
            raise NotImplementedError()

        
    def run_pure_auction(self):
        """
            Runs pure auction
        """

        # Iterate over all rounds ( R )
        for round_number in range(self.number_of_rounds):
            self.auction_results["Round {}".format(round_number)] = {}
            
            # List of all buyers at the beginning of each round_number
            buyer_indices = [i for i in range(self.buyer_prices.shape[0])]

            # Iterate over number of auctions which is the number of sellers as defined in the init function (same as K)
            for k in range(self.number_of_auctions):
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)] = {}

                logging.info('Sk value for current auction: {}'.format(self.seller_prices[k, round_number]))

                # Get buyer prices for k in K and r in R and for existing buyer indices
                # Buyer indices are reduced when one buyer wins an auction
                buyer_prices_for_auction = self.buyer_prices[buyer_indices, k, round_number]
                
                logging.info('Buyer Prices for the current auction: {}'.format(buyer_prices_for_auction))

                # Calculate Market Price
                market_price_for_auction = self.calculate_market_price(buyer_prices_for_auction)
                
                # Append to list to display later
                self.market_price_developments.append([market_price_for_auction, round_number * self.number_of_auctions + k])
                
                # Determine Auction Winner
                auction_winner = self.get_auction_winner(market_price_for_auction, buyer_prices_for_auction)

                # If an auction winner exists ( i.e if there is a second highest bid below Market Price)
                if len(auction_winner) != 0:

                    # Get seller profit
                    self.calculate_seller_profits(k, buyer_prices_for_auction[auction_winner[0]])
                    # Get buyer profit
                    self.calculate_buyer_profits(auction_winner[0], market_price_for_auction, buyer_prices_for_auction[auction_winner[0]])
                    # Remove buyer who wins the auction from being considered in the next auction in the same round
                    buyer_indices.pop(auction_winner[0])

                logging.info("Buyer Profits: {}".format(self.buyer_profits))
                logging.info("Seller Profits: {}".format(self.seller_profits))

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Winner"] = auction_winner
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Market Price"] = market_price_for_auction
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Buyer Profits"] = self.buyer_profits
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Seller Profits"] = self.seller_profits

    def calculate_market_price(self, prices):
        """
            Market price is average of all buyer prices in that auction
        """
        market_price = np.average(prices)
        logging.info("Market Price for current auction: {}".format(market_price))
        return market_price

    def calculate_seller_profits(self, seller, profits):
        """
            Add current auction profits to array of profits for a particular seller
        """
        self.seller_profits[seller] += profits

    def calculate_buyer_profits(self, buyer, market_price, profits):
        """
            Add current auction profits to array of profits for a particular buyer
        """
        self.buyer_profits[buyer] += market_price - profits

    def get_auction_winner(self, market_price, buyers_list):
        """
            Determine auction winner by getting all elements below the market price and selecting the second highest element among them.
            Incase second highest element doesn't exist, first highest is selected since there is no lower bid possible. 
        """
        prices_below_market = buyers_list < market_price
        auction_winner = np.where(buyers_list == np.partition(prices_below_market * buyers_list, -2)[-2])[0]
        if np.shape(auction_winner) == 0:
            auction_winner = np.argmax(prices_below_market * buyers_list)

        ## Possible case where no auction winner exists because the only single value below the market price,
        ## depends on the value of alpha
        logging.info("Auction Winner is Buyer: {}".format(auction_winner))
        return auction_winner


def prettyprintdict(dictionary):
    for key in dictionary:
        print("\n")
        print(key)
        for key_level2 in dictionary[key]:
            print("\n")
            print(key_level2)
            for key_level3 in dictionary[key][key_level2]:
                print(key_level3, dictionary[key][key_level2][key_level3])

        

    
if __name__ == "__main__":
    auction_input = AuctionCreator()
    auction_input.get_user_input()  
    seller_prices, buyer_prices = auction_input.create_bid_matrices()  
    auction_simulator = AuctionSimulator(seller_prices, buyer_prices, auction_input.auction_type)
    auction_simulator.run_auctions()
    prettyprintdict(auction_simulator.auction_results)
    print(auction_simulator.market_price_developments)
    ax = sns.lineplot(x='Rounds', y='Market Price', data=pd.DataFrame(auction_simulator.market_price_developments, columns=['Market Price', 'Rounds']))
    plt.show()