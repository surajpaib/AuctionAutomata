from auction_creator import AuctionCreator
import logging
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)

class AuctionSimulator:
    def __init__(self, seller_prices, buyer_prices, auction_type):
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
        if self.auction_type == 0:
            self.run_pure_auction()
        else:
            raise NotImplementedError()

        
    def run_pure_auction(self):
        for round in range(self.number_of_rounds):
            self.auction_results["Round {}".format(round)] = {}
            buyer_indices = [i for i in range(self.buyer_prices.shape[0])]


            for k in range(self.number_of_auctions):
                self.auction_results["Round {}".format(round)]["Auction {}".format(k)] = {}

                logging.info('Sk value for current auction: {}'.format(self.seller_prices[k, round]))
                buyer_prices_for_auction = self.buyer_prices[buyer_indices, k, round]
                
                logging.info('Buyer Prices for the current auction: {}'.format(buyer_prices_for_auction))
                market_price_for_auction = self.calculate_market_price(buyer_prices_for_auction)
                self.market_price_developments.append([market_price_for_auction, round * self.number_of_auctions + k])
                auction_winner = self.get_auction_winner(market_price_for_auction, buyer_prices_for_auction)

                if len(auction_winner) != 0:
                    self.calculate_seller_profits(k, buyer_prices_for_auction[auction_winner[0]])
                    self.calculate_buyer_profits(auction_winner[0], market_price_for_auction, buyer_prices_for_auction[auction_winner[0]])
                    print(buyer_indices)
                    print(auction_winner[0])
                    buyer_indices.pop(auction_winner[0])
                logging.info("Buyer Profits: {}".format(self.buyer_profits))
                logging.info("Seller Profits: {}".format(self.seller_profits))

                self.auction_results["Round {}".format(round)]["Auction {}".format(k)]["Winner"] = auction_winner
                self.auction_results["Round {}".format(round)]["Auction {}".format(k)]["Market Price"] = market_price_for_auction
                self.auction_results["Round {}".format(round)]["Auction {}".format(k)]["Buyer Profits"] = self.buyer_profits
                self.auction_results["Round {}".format(round)]["Auction {}".format(k)]["Seller Profits"] = self.seller_profits

    def calculate_market_price(self, prices):
        market_price = np.average(prices)
        logging.info("Market Price for current auction: {}".format(market_price))
        return market_price

    def calculate_seller_profits(self, seller, profits):
        self.seller_profits[seller] += profits

    def calculate_buyer_profits(self, buyer, market_price, profits):
        self.buyer_profits[buyer] += market_price - profits

    def get_auction_winner(self, market_price, buyers_list):
        prices_below_market = buyers_list < market_price
        auction_winner = np.where(buyers_list == np.partition(prices_below_market * buyers_list, -2)[-2])[0]

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