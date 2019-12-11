from auction_creator import AuctionCreator
import logging
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set level=None to remove display messages
logging.basicConfig(level=logging.INFO)


def get_participating_buyers(buyer_indices):
    buyers_list = []
    for key in buyer_indices:
        buyer = buyer_indices[key]
        if buyer['participating']:
            buyers_list.append(key)

    return buyers_list


class AuctionSimulator:
    def __init__(self, seller_prices, buyer_prices, alpha_factors, auction_type):
        """
            Create seller profits/ buyer profits array and define other variables 
        """
        self.auction_type = auction_type
        self.seller_prices = seller_prices
        self.buyer_prices = buyer_prices
        self.alpha_factors = alpha_factors

        self.seller_profits = np.zeros((self.seller_prices.shape[0]))
        self.buyer_profits = np.zeros((self.buyer_prices.shape[0]))
        self.number_of_buyers = self.buyer_prices.shape[0]
        self.number_of_rounds = self.buyer_prices.shape[2]
        self.number_of_auctions = self.seller_prices.shape[0]
        self.auction_results = {}
        self.market_price_developments = []
        self.bid_increase_factor = np.random.rand(self.number_of_buyers) * 0.1 + 1 #Greater than 1
        self.bid_decrease_factor = 1 - np.random.rand(self.number_of_buyers) * 0.1  #Lower than 1
        # print(self.bid_decrease_factor)
        # print(self.bid_increase_factor)
        # exit()

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

  
        for round_number in range(self.number_of_rounds):
            self.auction_results["Round {}".format(round_number)] = {}
            self.buyer_prices[:, :, round_number] = self.alpha_factors * self.seller_prices[:, round_number]
            # Bid increase or bid decrease for buyers based on win or above market price condition.
            if round_number > 0:
                self.adapt_bidding_strategy(buyer_indices, round_number)
            
            # List of all buyers at the beginning of each round_number

            buyer_indices = dict.fromkeys([i for i in range(self.buyer_prices.shape[0])], {'participating': True, 'bid_increase_sellers': [], 'bid_decrease_sellers': []})

            # Iterate over number of auctions which is the number of sellers as defined in the init function (same as K)
            for k in range(self.number_of_auctions):
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)] = {}

                logging.info('Sk value for current auction: {}'.format(self.seller_prices[k, round_number]))



                # Fetch Participating Buyers from Dict
                participating_buyers = get_participating_buyers(buyer_indices)
                print("Participating: ", participating_buyers)
                # Get buyer prices for k in K and r in R and for existing buyer indices
                self.buyer_prices_for_auction = self.buyer_prices[participating_buyers, k, round_number]
                logging.info("Alpha Factors for the buyers: {}".format(self.alpha_factors))
                logging.info('Buyer Prices for the current auction: {}'.format(self.buyer_prices_for_auction))

                # Calculate Market Price
                self.market_price_for_auction = self.calculate_market_price(self.buyer_prices_for_auction)
                
                # Append to list to display later
                self.market_price_developments.append([self.market_price_for_auction, round_number * self.number_of_auctions + k])
                
                # Determine Auction Winner
                self.participating_buyers_index = self.get_auction_winner(self.market_price_for_auction, self.buyer_prices_for_auction)
                self.auction_winner_index = participating_buyers[self.participating_buyers_index]

                # Get seller profit
                self.calculate_seller_profits(k, self.buyer_prices_for_auction[self.participating_buyers_index])
                # Get buyer profit
                self.calculate_buyer_profits(self.auction_winner_index, self.market_price_for_auction, self.buyer_prices_for_auction[self.participating_buyers_index])
                # Remove buyer who wins the auction from being considered in the next auction in the same round and decrease their bid. 
                buyer_indices = self.set_bidding_strategy(buyer_indices, k)
                print("Updated Buyers: ", buyer_indices)
                logging.info("Auction Winner is Buyer: {}".format(self.auction_winner_index))

                logging.info("Buyer Profits: {}".format(self.buyer_profits))
                logging.info("Seller Profits: {}".format(self.seller_profits))

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Winner"] = self.auction_winner_index
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Market Price"] = self.market_price_for_auction
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Buyer Profits"] = self.buyer_profits
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Seller Profits"] = self.seller_profits

    

    def set_bidding_strategy(self, buyer_indices, k):
        """
        Sets the buyers increase_bid flag based on winning an auction or bidding above market price
        """

        
        buyer_indices[self.auction_winner_index] =  {'participating': False, 'bid_increase_sellers': buyer_indices[self.auction_winner_index]['bid_increase_sellers'] , 'bid_decrease_sellers': buyer_indices[self.auction_winner_index]['bid_decrease_sellers']+ [k]}
        
        for index in np.where(self.buyer_prices_for_auction >= self.market_price_for_auction)[0]:
            if buyer_indices[index]['participating'] == True:

                buyer_indices[index] =  {'participating': True, 'bid_increase_sellers': buyer_indices[index]['bid_increase_sellers'] , 'bid_decrease_sellers': buyer_indices[index]['bid_decrease_sellers']+ [k]}
            
        for index in np.where(self.buyer_prices_for_auction < self.market_price_for_auction)[0]:
            if buyer_indices[index]['participating'] == True:
                buyer_indices[index] =  {'participating': True, 'bid_increase_sellers': buyer_indices[index]['bid_increase_sellers']+ [k], 'bid_decrease_sellers': buyer_indices[index]['bid_decrease_sellers'] }
            
        return buyer_indices
        

    def adapt_bidding_strategy(self, buyer_indices, round_number):
        print("Adapting Bidding Strategy ...")
        for key in buyer_indices:
            buyer = buyer_indices[key]
            for k in buyer['bid_increase_sellers']:
                self.alpha_factors[key, k] = self.alpha_factors[key, k]  * self.bid_increase_factor[key] 

            for k in buyer['bid_decrease_sellers']:
                self.alpha_factors[key, k] =  self.alpha_factors[key, k]  * self.bid_decrease_factor[key] 


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


        if np.sum(prices_below_market) > 1:
            auction_winner = np.where(buyers_list == np.partition(prices_below_market * buyers_list, -2)[-2])[0][0]

        else:
  
            auction_winner = np.argmax(prices_below_market * buyers_list)
        ## Possible case where no auction winner exists because the only single value below the market price,
        ## depends on the value of alpha
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
    seller_prices, buyer_prices, alpha_factors = auction_input.create_bid_matrices()  
    auction_simulator = AuctionSimulator(seller_prices, buyer_prices, alpha_factors, auction_input.auction_type)
    auction_simulator.run_auctions()
    # prettyprintdict(auction_simulator.auction_results)
    # print(auction_simulator.market_price_developments)
    ax = sns.lineplot(x='Rounds', y='Market Price', data=pd.DataFrame(auction_simulator.market_price_developments, columns=['Market Price', 'Rounds']))
    # plt.show()