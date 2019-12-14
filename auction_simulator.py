from auction_creator import AuctionCreator
import logging
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

# Set level=None to remove display messages
logging.basicConfig(level=logging.INFO)



class AuctionSimulator:
    def __init__(self, seller_prices, buyer_prices, alpha_factors, epsilon, auction_type):
        """
            Create seller profits/ buyer profits array and define other variables 
        """
        self.auction_type = auction_type
        self.seller_prices = seller_prices
        self.buyer_prices = buyer_prices
        self.epsilon = epsilon
        self.alpha_factors = alpha_factors


        self.seller_profits = np.zeros((self.seller_prices.shape[0]))
        self.buyer_profits = np.zeros((self.buyer_prices.shape[0]))
        self.number_of_buyers = self.buyer_prices.shape[0]
        self.number_of_rounds = self.buyer_prices.shape[2]
        self.number_of_auctions = self.seller_prices.shape[0]
        self.auction_results = {}
        self.market_price_developments = []

        #self.bid_increase_factor = np.linspace(1, 1.1, num=self.number_of_buyers)
        #self.bid_decrease_factor = np.linspace(1, 0.9, num=self.number_of_buyers)
        self.bid_increase_factor = np.random.rand(self.number_of_buyers) * 0.1 + 1
        self.bid_decrease_factor = 1 - np.random.rand(self.number_of_buyers) * 0.1
        #print(self.bid_decrease_factor)
        #print(self.bid_increase_factor)


    def run_auctions(self):
        """
            Run pure or levelled commitment auction
        """
        if self.auction_type == 0:
            self.run_pure_auction()
        else:
            self.run_levelled_commitment_auction()

    def run_levelled_commitment_auction(self):

        for round_number in range(self.number_of_rounds):
            self.auction_results["Round {}".format(round_number)] = {}
            self.alpha_factors[self.alpha_factors < 1.0] = 1.0

            # Boolean Array to store previous bids by a buyer. True if bid at n,k
            self.commited_buyer_bids = np.zeros((self.number_of_buyers, self.number_of_auctions), dtype=bool)
            
            # Buyer and Seller Profits for the Round
            buyer_profits = np.zeros((self.number_of_buyers, self.number_of_auctions))
            seller_profits = np.zeros(self.number_of_auctions)

            # Buyer profits including annuled auctions
            total_buyer_profits =  np.zeros((self.number_of_buyers, self.number_of_auctions))
            
            # Buyer prices for the round adapted with alpha factors
            self.buyer_prices[:, :, round_number] = self.alpha_factors * self.seller_prices[:, round_number]


            for k in range(self.number_of_auctions):
                buyer_indices = [i for i in range(self.buyer_prices.shape[0])]

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)] = {}

                logging.info('Sk value for current auction: {}'.format(self.seller_prices[k, round_number]))
                logging.info('Alpha Values for current auction: {}'.format(self.alpha_factors[:, k]))
                buyer_prices_for_auction = self.buyer_prices[:, k, round_number]
                logging.info('Buyer Prices befor Annuling: {}'.format(buyer_prices_for_auction))
                
                # Get previous auctions won for each buyer
                previous_auctions_won = np.argwhere(self.commited_buyer_bids == True)
                print(previous_auctions_won)

                # Set new bids based on the bidding strategy for buyers that have won previous auctions considering the deducted penalty and profit from previous auction.
                for buyer_index, x in previous_auctions_won:
                    buyer_prices_for_auction[buyer_index] =  self.buyer_prices[buyer_index, k, round_number] - buyer_profits[buyer_index, x] - self.epsilon * self.buyer_prices[buyer_index, x, round_number]
                    if buyer_prices_for_auction[buyer_index] < self.seller_prices[k, round_number]:
                        buyer_indices.remove(buyer_index)
                
                buyer_prices_for_auction = self.buyer_prices[buyer_indices, k, round_number]

                logging.info("Alpha Factors for the buyers: {}".format(self.alpha_factors))
                logging.info('Buyer Prices for the current auction: {}'.format(buyer_prices_for_auction))

                # Calculate Market Price
                market_price_for_auction = self.calculate_market_price(buyer_prices_for_auction)
                
                # Append to list to display later
                self.market_price_developments.append([market_price_for_auction, round_number * self.number_of_auctions + k])
               
                # Determine Auction Winner
                auction_winner_index = self.get_auction_winner(market_price_for_auction, buyer_prices_for_auction)
                auction_winner = buyer_indices[auction_winner_index]
                logging.info("Auction Winner is Buyer: {}".format(auction_winner))   

                # Set auction winner in the commited bids
                self.commited_buyer_bids[auction_winner, k] = True


                # Update Buyer Profits for this auction; Not sure if this should have the penalty included because the bid already considers the penalty?
                winning_bid =  self.get_winning_bid(buyer_prices_for_auction, auction_winner_index)
                buyer_profits[auction_winner, k] = market_price_for_auction - winning_bid

                logging.info('Buyer Profits for Auction: ( May get annuled later): {}'.format(buyer_profits))
                # Update seller profits for this auction
                seller_profits[k] = winning_bid
                total_buyer_profits[auction_winner, k] = buyer_profits[auction_winner, k]
                # Annuling an auction, check for all previous won auctions for the buyer
                for buyer_index, x in previous_auctions_won:
                    if buyer_index == auction_winner:
                        # If the buyer makes a better profit on this auction compared to the previous won auction then it is more profitable for the buyer, the bid includes the penalty and the profit
                        if buyer_profits[auction_winner, k]  > buyer_profits[auction_winner, x]:
                            self.commited_buyer_bids[auction_winner, x] = False

                            # Buyer profits for this auction involve a penalty to be paid for annuling the x auction.
                            total_buyer_profits[auction_winner, k] = buyer_profits[auction_winner, k] - self.epsilon *  self.buyer_prices[buyer_index, x, round_number]

                            # Seller profits for the annuled round is the penalty.
                            seller_profits[x] = self.epsilon *  self.buyer_prices[buyer_index, x, round_number]


            # Sum seller profits across all rounds
            self.seller_profits += seller_profits 

            # The commited buyer bids is transposed to obtain auction winners across each round x-> round
            for x, auction_winner in np.argwhere(self.commited_buyer_bids.T):
                logging.info("Auction Winner is Buyer: {}".format(auction_winner))   
                # Buyer profits consider only auction winners for non-annuled rounds  
                self.buyer_profits[auction_winner] += total_buyer_profits[auction_winner, x] 

                buyer_prices_for_auction = self.buyer_prices[:, x, round_number]
                market_price_for_auction = self.calculate_market_price(buyer_prices_for_auction)

                # Bid increase and bid decrease for the auctions in the round ( only for non-annuled auctions)
                self.adapt_alpha_factors(market_price_for_auction, buyer_prices_for_auction, [i for i in range(self.buyer_prices.shape[0])], auction_winner, x)

                logging.info("Buyer Profits: {}".format(self.buyer_profits))
                logging.info("Seller Profits: {}".format(self.seller_profits))

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(x)]["Winner"] = auction_winner
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(x)]["Market Price"] = market_price_for_auction
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(x)]["Buyer Profits"] = self.buyer_profits
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(x)]["Seller Profits"] = self.seller_profits


    def run_pure_auction(self):
        """
            Runs pure auction
        """

  
        for round_number in range(self.number_of_rounds):
            self.auction_results["Round {}".format(round_number)] = {}
            self.alpha_factors[self.alpha_factors < 1.0] = 1.0
            self.buyer_prices[:, :, round_number] = self.alpha_factors * self.seller_prices[:, round_number]
            # List of all buyers at the beginning of each round_number
            buyer_indices = [i for i in range(self.buyer_prices.shape[0])]

            # Iterate over number of auctions which is the number of sellers as defined in the init function (same as K)
            for k in range(self.number_of_auctions):

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)] = {}

                logging.info('Sk value for current auction: {}'.format(self.seller_prices[k, round_number]))
                logging.info('Alpha Values for current auction: {}'.format(self.alpha_factors[:, k]))
                # Get buyer prices for k in K and r in R and for existing buyer indices
                buyer_prices_for_auction = self.buyer_prices[buyer_indices, k, round_number]
                logging.info("Alpha Factors for the buyers: {}".format(self.alpha_factors))
                logging.info('Buyer Prices for the current auction: {}'.format(buyer_prices_for_auction))

                # Calculate Market Price
                market_price_for_auction = self.calculate_market_price(buyer_prices_for_auction)
                
                # Append to list to display later
                self.market_price_developments.append([market_price_for_auction, round_number * self.number_of_auctions + k])
                # Determine Auction Winner
                auction_winner_index = self.get_auction_winner(market_price_for_auction, buyer_prices_for_auction)
                auction_winner = buyer_indices[auction_winner_index]

                logging.info("Auction Winner is Buyer: {}".format(auction_winner))     

                # Get seller profit
                self.calculate_seller_profits(k, buyer_prices_for_auction, auction_winner_index)
                # Get buyer profit
                self.calculate_buyer_profits(auction_winner, market_price_for_auction, buyer_prices_for_auction, auction_winner_index)
                # Remove buyer who wins the auction from being considered in the next auction in the same round

                logging.info("Buyer Profits: {}".format(self.buyer_profits))
                logging.info("Seller Profits: {}".format(self.seller_profits))

                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Winner"] = auction_winner
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Market Price"] = market_price_for_auction
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Buyer Profits"] = self.buyer_profits
                self.auction_results["Round {}".format(round_number)]["Auction {}".format(k)]["Seller Profits"] = self.seller_profits


                self.adapt_alpha_factors(market_price_for_auction, buyer_prices_for_auction, buyer_indices, auction_winner, k)
                
                buyer_indices.pop(auction_winner_index)


    def adapt_alpha_factors(self, market_price_for_auction, buyer_prices_for_auction, buyer_indices, auction_winner, k):
        buyers_above_market_price = [buyer_indices[i] for i in np.where(buyer_prices_for_auction >= market_price_for_auction)[0]]
        buyers_decrease_bid = buyers_above_market_price + [auction_winner]
        buyers_increase_bid =  list(set(buyer_indices) - set(buyers_decrease_bid))
        print(self.bid_decrease_factor[buyers_decrease_bid] * self.alpha_factors[buyers_decrease_bid, k])
        self.alpha_factors[buyers_decrease_bid, k] = self.bid_decrease_factor[buyers_decrease_bid] * self.alpha_factors[buyers_decrease_bid, k]
        self.alpha_factors[buyers_increase_bid, k] = self.bid_increase_factor[buyers_increase_bid] * self.alpha_factors[buyers_increase_bid, k]


    def calculate_market_price(self, prices):
        """
            Market price is average of all buyer prices in that auction
        """
        market_price = np.average(prices)
        logging.info("Market Price for current auction: {}".format(market_price))
        return market_price


    def get_winning_bid(self, buyer_prices, auction_winner_index):

        second_winning_bid = buyer_prices < buyer_prices[auction_winner_index]

        if np.sum(second_winning_bid) >= 1:
            winning_bid = np.max(buyer_prices * second_winning_bid)

        else:
            winning_bid = buyer_prices[auction_winner_index]

        return winning_bid

    def calculate_seller_profits(self, seller, buyer_prices, auction_winner_index):
        """
            Add current auction profits to array of profits for a particular seller
        """
    
        winning_bid = self.get_winning_bid(buyer_prices, auction_winner_index)
        self.seller_profits[seller] += winning_bid

    def calculate_buyer_profits(self, buyer, market_price, buyer_prices, auction_winner_index):
        """
            Add current auction profits to array of profits for a particular buyer
        """
        winning_bid = self.get_winning_bid(buyer_prices, auction_winner_index)
        self.buyer_profits[buyer] += market_price - winning_bid

    def get_auction_winner(self, market_price, buyers_list):
        """
            Determine auction winner by getting all elements below the market price and selecting the second highest element among them.
            Incase second highest element doesn't exist, first highest is selected since there is no lower bid possible. 
        """
        prices_below_market = buyers_list < market_price

        auction_winner = np.argmax(prices_below_market * buyers_list)
     
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
    auction_input.create_bid_matrices()  
    auction_simulator = AuctionSimulator(auction_input.seller_price, auction_input.buyer_price, auction_input.alpha_factors, auction_input.epsilon, auction_input.auction_type)
    auction_simulator.run_auctions()
    # prettyprintdict(auction_simulator.auction_results)
    plt.figure()
    ax = sns.scatterplot(x='Rounds', y='Market Price', data=pd.DataFrame(auction_simulator.market_price_developments, columns=['Market Price', 'Rounds']))

    price_variance = []

    for sellers in range(auction_simulator.number_of_auctions):
        for rounds in range(auction_simulator.number_of_rounds):

            price_variance.append([np.var(auction_simulator.buyer_prices[:, sellers, rounds]), rounds, sellers])
    
    plt.figure()


    px = sns.lineplot(x='Rounds', y='Variance of Buyer Prices', hue='Seller', data=pd.DataFrame(price_variance, columns=['Variance of Buyer Prices', 'Rounds', 'Seller']))

    ls = price_variance[(auction_simulator.number_of_auctions*auction_simulator.number_of_rounds-auction_simulator.number_of_auctions):auction_simulator.number_of_auctions*auction_simulator.number_of_rounds]
    m = np.mean(ls, axis=0)
    s_profits = np.mean(auction_simulator.seller_profits)
    b_profits = np.mean(auction_simulator.buyer_profits)
    print(m)
    print(b_profits)
    print(s_profits)

    plt.show()
