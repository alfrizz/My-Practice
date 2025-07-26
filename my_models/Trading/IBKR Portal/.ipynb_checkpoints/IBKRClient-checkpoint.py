import requests
import json
import urllib3
import time
import os
import subprocess
import webbrowser
import websocket
import ssl
from datetime import datetime


class IBKRClient:

    def __init__(self, 
                 account_id,
                 base_url="https://localhost:5000/v1/api",  
                 dir_path=r"C:\IBKR Client Portal", 
                 verify_ssl=False):
        """
        Initializes the IBKR API client with configuration parameters.
        Runs the IBKR client login command only if authentication is not already established.
        """
        self.base_url = base_url.rstrip("/")
        self.account_id = account_id
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
        # Check authentication status and perform login if needed...
        auth_endpoint = "iserver/auth/status"
        auth_url = f"{self.base_url}/{auth_endpoint}"
        print("Checking authentication status at:", auth_url)
        try:
            auth_req = self.session.get(url=auth_url, verify=self.verify_ssl, timeout=3)
        except Exception as e:
            print("Error checking authentication:", e)
            auth_req = None
        if auth_req is not None and auth_req.status_code == 200:
            try:
                data = auth_req.json()
                if data.get("authenticated") == True:
                    print("Already logged in:", data)
                else:
                    print("Received 200 but not authenticated:", data)
                    self.login(dir_path, auth_url)
            except Exception as e:
                print("Error parsing auth response:", e)
        else:
            self.login(dir_path, auth_url)

            

    def login(self, dir_path, auth_url):
        print("Not logged in. Initiating login process.")
        os.chdir(dir_path)
        print(f"Changed directory to: {dir_path}")
        login_command = r"bin\run.bat root\conf.yaml"
        process = subprocess.Popen(
            login_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print("Command Output:", stdout)
            if stderr:
                print("Error Output:", stderr)
        except subprocess.TimeoutExpired:
            print("The process is running... (output not complete yet).")
        url = "https://localhost:5000"
        print(f"Opening IBKR Client at: {url}")
        webbrowser.open(url)
        print("Polling authentication status at:", auth_url)
        while True:
            try:
                auth_req = self.session.get(url=auth_url, verify=self.verify_ssl)
                if auth_req.status_code == 200:
                    data = auth_req.json()
                    if data.get("authenticated") == True:
                        print("Authentication successful:", data)
                        break
                    else:
                        print("Received 200 but not authenticated yet:", data)
                else:
                    print("Received status code:", auth_req.status_code)
            except Exception as exc:
                print("Error during auth status polling:", exc)
            time.sleep(5)

        

    def contract_search(self, symbol, secType=False):
        """
        Searches for contract details using the Interactive Brokers API.
        Sends a POST request with search parameters.
        
        If a matching contract is found, this method extracts the first contract's 
        "conid" and its ticker value (returned as the symbol) and returns them as a tuple.
        
        Parameters:
            symbol (str): The stock or contract symbol (e.g., "ES").
            secType (str or bool): The type of security (e.g., "STK"). Defaults to False.
        
        Returns:
            tuple or None: (conid, symbol) if a matching contract is found; otherwise, None.
        """
        endpoint = "iserver/secdef/search"
        url = f"{self.base_url}/{endpoint}"
        print("Contract Search URL:", url)
        
        json_body = {
            "symbol": symbol,
            "secType": secType,
            "name": False
        }
        
        try:
            response = self.session.post(url=url, verify=self.verify_ssl, json=json_body)
            print("Response Status Code:", response.status_code)
            
            data = response.json()
            print("Contract Search Results:")
            print(json.dumps(data, indent=2))
            
            if isinstance(data, list) and len(data) > 0 and "conid" in data[0]:
                first_contract = data[0]
                first_conid = first_contract["conid"]
                first_ticker = first_contract.get("symbol", symbol)
                print(f"First conid found: {first_conid}")
                print(f"Ticker returned as symbol: {first_ticker}")
                return first_conid, first_ticker  # Return a tuple instead of a dictionary.
            else:
                print("No valid contract found in the response.")
                return None
        except Exception as e:
            print("Error searching for contract details:", e)
            return None
            

    
    def contract_info(self, conid):
        """
        Retrieves specific contract information from the API using a given contract ID.
        
        Key Features:
          - Builds the request URL by combining the base URL, endpoint, and query parameters.
          - Sends a GET request to the contract information endpoint.
          - Prints the HTTP response details and a formatted JSON output.
          - Returns the contract information as a dictionary.
        
        Parameters:
            conid (str, int): The contract ID of the instrument (e.g., "265598").
        
        Returns:
            dict or None: A dictionary containing the contract details if the request is successful;
                          otherwise, None.
        """
        endpoint = "iserver/secdef/info"
        
        # Build query parameters string.
        # If needed, you can include more parameters (e.g., secType, month, exchange, strike, right)
        # For now, we'll simply use the contract id parameter.
        param_str = f"conid={conid}"
        # To add other parameters, you could do:
        # param_str = "&".join([
        #    f"conid={conid}",
        #    "secType=FOP",
        #    "month=JUL25",
        #    "exchange=CME",
        #    "strike=4800",
        #    "right=C"
        # ])
        
        # Construct the full URL.
        request_url = f"{self.base_url}/{endpoint}?{param_str}"
        print("Contract Info Request URL:", request_url)
        
        try:
            # Perform the GET request.
            response = self.session.get(url=request_url)
            print("HTTP Response:", response)
            
            # Parse the response as JSON.
            contract_details = response.json()
            # Convert to formatted JSON string for printing.
            contract_json = json.dumps(contract_details, indent=2)
            print("Contract Info:")
            print(contract_json)
            
            return contract_details
        except Exception as e:
            print("Error retrieving contract info:", e)
            return None



    def historical_data(self, contract_id, period="1w", bar="1d", outsideRth="true", barType="midpoint"):
        """
        Retrieves historical data for one or multiple contract IDs.
    
        Parameters:
            contract_id (int, str): A single contract ID
            period (str): Historical period, e.g., "1w" for one week.
            bar (str): Time resolution, e.g., "1d" for one day bars.
            outsideRth (str): Whether to include data outside regular trading hours.
            barType (str): Type of bar (e.g., "midpoint").
    
        Returns:
            dict or None: The historical data if available; otherwise, None.
        """
        endpoint = "hmds/history"
        params = f"conid={contract_id}&period={period}&bar={bar}&outsideRth={outsideRth}&barType={barType}"
        request_url = f"{self.base_url}/{endpoint}?{params}"
        print("Historical Data Request URL:", request_url)
    
        max_retries = 3
        for attempt in range(max_retries):
            response = self.session.get(url=request_url)
            print("HTTP Status Code:", response.status_code)
            try:
                data = response.json()
                print("Historical Data:", json.dumps(data, indent=2))
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding historical data JSON on attempt {attempt+1}: {e}")
                print("Raw Response:", response.text)
                # If we're not on the last attempt, wait and retry.
                if attempt < max_retries - 1:
                    print("Retrying after a short delay...")
                    time.sleep(3)
                else:
                    print("Max retries reached. Returning None.")
                    return None

    
    
    def market_snapshot(self, conids_by_comas):
        """
        Retrieves live market data snapshots for the given contract IDs and extracts the last price.
        
        Process:
          1. Constructs query parameters including the comma-separated contract IDs and specific data fields.
          2. Constructs the full request URL using the base URL, the market snapshot endpoint, and the query parameters.
          3. Continuously sends a GET request to the API endpoint until the 'last price' (field "31") is retrieved.
          4. Parses and prints the JSON response and extracts the value for field "31" as a float.
        
        Parameters:
            conids_by_comas (str): A string of contract IDs separated by commas (e.g., "11111,22222").
        
        Returns:
            float: The last price extracted from field "31" from the first element of the returned data.
        """
        endpoint = "iserver/marketdata/snapshot"
        # Specify the desired data fields:
        # 31 (last price), 84 (bid price), 86 (ask price), 7768 (trading permissions).
        fields = "fields=31,84,86,7768"
        conids = f"conids={conids_by_comas}"
        
        # Combine the parameters into one query string.
        params = "&".join([conids, fields])
        print("Query Parameters:", params)
        
        # Build the full request URL.
        request_url = f"{self.base_url}/{endpoint}?{params}"
        print("Market Snapshot Request URL:", request_url)
        
        last_price = None
        while last_price is None:
            try:
                md_req = self.session.get(url=request_url, verify=self.verify_ssl)
                print("HTTP Status Code:", md_req.status_code)
                data = md_req.json()
                md_json = json.dumps(data, indent=2)
                print("Market Snapshot Data:")
                print(md_json)
                
                if data and len(data) > 0:
                    last_price_str = data[0].get("31")
                    if last_price_str is not None:
                        last_price = float(last_price_str)
                        print("Extracted Last Price:", last_price)
                    else:
                        print("Field '31' not found in the snapshot data. Retrying...")
                else:
                    print("Empty data received. Retrying...")
            except Exception as e:
                print("Error fetching market snapshot:", e)
            
            if last_price is None:
                print("Waiting before next polling attempt...")
                time.sleep(3)
        
        return last_price


    

    def contract_strikes(self, conid, month, secType, exchange):
        """
        Retrieves available strike prices for a given contract.

        This method builds a GET request URL using the provided parameters, sends the request,
        and then returns the strike price details in JSON format.

        Parameters:
            conid (str or int): The contract identifier (e.g., "11004968").
            secType (str, optional): Security type
            month (str, optional): Expiry month
            exchange (str, optional): Exchange where the contract is traded.

        Returns:
            dict or None: The JSON response containing strike price details if successful; otherwise, None.
        """
        endpoint = "iserver/secdef/strikes"
        # Construct the query string from the parameters.
        params = f"conid={conid}&secType={secType}&month={month}&exchange={exchange}"
        
        # Build the full request URL.
        request_url = f"{self.base_url}/{endpoint}?{params}"
        print("Strike Prices Request URL:", request_url)
        
        try:
            # Perform the GET request to fetch strike price information.
            response = self.session.get(url=request_url, verify=self.verify_ssl)
            print("HTTP Status Code:", response.status_code)
            
            # Parse and pretty print the JSON response.
            data = response.json()
            print("Strike Price Data:")
            print(json.dumps(data, indent=2))
            return data
        except Exception as e:
            print("Error fetching strike prices:", e)
            return None


    
    
    def order_request(self, conid, orderType, side, quantity, tif, price=False):
        """
        Sends an order request to the API and extracts the order 'id' from the response.
        
        For stop orders (orderType == "STP"), the 'price' parameter is required.
        
        Parameters:
            conid (int or str): Contract ID for the instrument.
            orderType (str): Order type (e.g., "STP" for stop orders, "MKT" for market orders).
            side (str): Order side (e.g., "Buy" or "Sell").
            quantity (int or str): Number of units to buy or sell.
            tif (str): Time in Force (e.g., "DAY", "GTC").
            price (float, optional): The price value for the order. This parameter is required if orderType is "STP".
        
        Returns:
            str: The extracted order id from the response.
        
        Raises:
            ValueError: If a stop order is requested without a valid price.
        """
        # Build the order entry.
        order_entry = {
            "conid": conid,
            "orderType": orderType,
            "side": side,
            "tif": tif,
            "quantity": quantity
        }
        
        # If it's a stop order, ensure that a price is provided.
        if orderType.upper() == "STP":
            if not price:
                raise ValueError("For stop orders, the 'price' parameter must be provided.")
            order_entry["price"] = price
        
        json_body = {"orders": [order_entry]}
        print("Order Request JSON Body:")
        print(json.dumps(json_body, indent=2))
        
        # Construct the full request URL using the stored account_id.
        endpoint = f"iserver/account/{self.account_id}/orders"
        request_url = f"{self.base_url}/{endpoint}"
        print("Order Request URL:", request_url)
        
        # Send the POST request.
        order_req = self.session.post(url=request_url, verify=self.verify_ssl, json=json_body)
        print("HTTP Status Code:", order_req.status_code)
        
        # Parse and print the response.
        order_data = order_req.json()
        print("Response Data:", json.dumps(order_data, indent=2))
        
        # Extract and return the order ID from the first element.
        reply_id = order_data[0]["id"]
        return reply_id

    

    
    def order_reply(self, initial_reply_id):
        """
        Sends confirmation replies for a given order reply id and polls the endpoint
        until no additional (new) reply id is provided.
        
        In other words, if the API returns a new reply id, then a further confirmation
        (another POST) is required. When the response does not include a new reply id
        (or it repeats the current id), the polling stops.
        
        Parameters:
            initial_reply_id (str): The starting reply identifier.
        
        Returns:
            str or None: The latest reply id if the confirmation process ended naturally;
                         None if no new reply id was ever provided.
        """
        current_reply_id = initial_reply_id
        json_body = {"confirmed": True}
    
        while True:
            endpoint = f"iserver/reply/{current_reply_id}"
            request_url = f"{self.base_url}/{endpoint}"
            print("Posting confirmation to URL:", request_url)
            response = self.session.post(url=request_url, json=json_body)
            print("Status Code:", response.status_code)
            print("Raw Response:", response.text)
    
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print("Error decoding JSON response:", e)
                break  # Or return current_reply_id if you prefer.
    
            # Ensure we got a list response with at least one element.
            if isinstance(data, list) and data:
                new_reply_id = data[0].get("id", None)
                
                # If there are any confirmation messages/questions, print them.
                messages = data[0].get("message", [])
                if messages:
                    print("Additional messages received:")
                    for msg in messages:
                        print(msg)
                
                # If no new reply id is provided, break out.
                if not new_reply_id:
                    print("No reply id provided in response. Breaking out.")
                    break
                # If the new reply id is different from the current one,
                # update and continue polling.
                if new_reply_id != current_reply_id:
                    print(f"New reply id received: {new_reply_id}")
                    current_reply_id = new_reply_id
                    # Continue polling with the new reply id.
                else:
                    print("Reply id hasn't changed. No further confirmation needed. Breaking out.")
                    break
            else:
                print("Response format unexpected or no data returned. Breaking out.")
                break
    
            time.sleep(3)  # Wait before the next poll.
    
        return current_reply_id


    def get_orders(self, ticker=False):
        """
        Retrieves all orders placed for the account using the endpoint "iserver/account/orders".
        If a ticker value is provided (different than False), filters the orders to only
        those matching the provided ticker.
        
        Process:
          1. Constructs the full URL and sends a GET request to the orders endpoint.
          2. Continuously polls until valid orders data is retrieved.
          3. If a ticker is provided, filters the orders based on the "ticker" field.
        
        Parameters:
            ticker (str or bool): The ticker symbol to filter orders by (default is False,
                                    meaning no filtering).
        
        Returns:
            list or dict or None: The list of orders (filtered if a ticker is provided) if successful,
                                  otherwise None.
        """
        endpoint = "iserver/account/orders"
        request_url = f"{self.base_url}/{endpoint}"
        print("Orders Request URL:", request_url)
        
        orders_data = None
        # Polling loop to ensure orders data is retrieved.
        while orders_data is None:
            try:
                response = self.session.get(url=request_url, verify=self.verify_ssl)
                print("HTTP Status Code:", response.status_code)
                if response.status_code != 200:
                    print("Non-200 status code received. Polling again in 3 seconds...")
                    time.sleep(3)
                    continue
                
                data = response.json()
                # Assuming the orders are returned under the key "orders".
                if isinstance(data, dict) and "orders" in data:
                    orders_data = data["orders"]
                else:
                    orders_data = data
            except Exception as e:
                print("Error fetching orders:", e)
                print("Polling again in 3 seconds...")
                time.sleep(3)
        
        # If a ticker filter is applied, filter the orders.
        if ticker:
            filtered_orders = [order for order in orders_data if order.get("ticker") == ticker]
            print(f"Filtered Orders for ticker '{ticker}':")
            print(json.dumps(filtered_orders, indent=2))
            return filtered_orders
        else:
            print("All Orders Data:")
            print(json.dumps(orders_data, indent=2))
            return orders_data


    def order_modify(self, order_id, conid, orderType, price, side, tif, quantity):
        """
        Modifies an existing order with the provided details, and returns the new order id.

        The method normalizes the input parameters (e.g., converts tif to uppercase
        and converts orderType 'Stop' to 'STP') and posts the updated order details.
        
        Parameters:
            order_id (str): The identifier of the order to modify.
            conid (int or str): The instrument's contract ID.
            orderType (str): The order type (e.g., "Stop" will be converted to "STP",
                             "Market" could be converted to "MKT", etc.).
            price (float): The new price for the order.
            side (str): The side of the order, e.g., "BUY" or "SELL".
            tif (str): Time In Force (e.g., "Day" will be converted to "DAY", "GTC" remains as is).
            quantity (float or int): The quantity for the order.
        
        Returns:
            str or None: The new order identifier if retrieved; otherwise, None.
        """
        # Normalize inputs.
        tif = tif.upper()
        if orderType.lower() == "stop":
            orderType = "STP"
        elif orderType.lower() == "market":
            orderType = "MKT"
        
        side = side.upper()  # Ensure the side is in uppercase.
        
        # Construct the modify URL.
        endpoint = f"iserver/account/{self.account_id}/order/"
        modify_url = f"{self.base_url}/{endpoint}{order_id}"
        print("Modify Order URL:", modify_url)
        
        # Build the JSON payload.
        json_body = {
            "conid": conid,
            "orderType": orderType,
            "price": price,
            "side": side,
            "tif": tif,
            "quantity": quantity
        }
        print("Order Modify JSON Body:")
        print(json.dumps(json_body, indent=2))
        
        # Send the POST request to modify the order.
        order_req = self.session.post(url=modify_url, verify=self.verify_ssl, json=json_body)
        print("HTTP Status Code:", order_req.status_code)
        
        try:
            order_response = order_req.json()
            print("Response Data:")
            print(json.dumps(order_response, indent=2))
        except Exception as e:
            print("Error parsing response:", e)
            return None
        
        # Extract and return the new order id from the first element in the response.
        if isinstance(order_response, list) and len(order_response) > 0:
            new_id = order_response[0].get("id")
            if new_id:
                print("Extracted Order ID:", new_id)
                return new_id
            else:
                print("No order id found in the response.")
                return None
        else:
            print("Unexpected response format:", order_response)
            return None



    def order_cancel(self, order_id):
        """
        Cancels an order based on the given order identifier.

        Parameters:
            order_id (str): The identifier of the order to cancel.

        Process:
            1. Constructs the cancellation URL using the stored account id and base URL.
            2. Sends an HTTP DELETE request to cancel the order.
            3. Parses, prints, and returns the JSON response from the API.

        Returns:
            dict or None: The parsed JSON response containing cancellation details;
                          None if an error occurs.
        """
        # Construct the endpoint path incorporating the account ID.
        endpoint = f"iserver/account/{self.account_id}/order/"
        cancel_url = f"{self.base_url}/{endpoint}{order_id}"
        print("Order Cancel URL:", cancel_url)
        
        try:
            cancel_req = self.session.delete(url=cancel_url, verify=self.verify_ssl)
            print("HTTP Status Code:", cancel_req.status_code)
            # Try to parse the JSON response.
            response_data = cancel_req.json()
            formatted = json.dumps(response_data, indent=2)
            print("Response Data:")
            print(formatted)
            return response_data
        except Exception as e:
            print("Error cancelling order or parsing response:", e)
            return None
    

    
    def order_bracket(self,
                      ticker,
                      conid,
                      quantity,
                      primary_side="BUY",
                      primary_order_type="MKT",
                      profit_target_price=None,       # e.g. 157.00 for profit target
                      profit_order_type="LMT",
                      stop_loss_price=None,           # e.g. 157.30 for stop loss
                      stop_order_type="STP",
                      tif="GTC",
                      listing_exchange="SMART",
                      outside_rth=False,
                      referrer="QuickTrade",
                      is_single_group=False):         # Optional flag for OCA groups
        """
        Creates and sends a bracket order matching Interactive Brokers' documentation.
        
        The payload consists of a parent (primary) order and two child orders (profit taker and stop loss),
        where the parent order includes a 'cOID' field and the children include a 'parentId' field.
        
        Parameters:
          ticker (str): Ticker symbol.
          conid (int): The contract id of the underlying instrument.
          quantity (int): Number of shares/contracts.
          primary_side (str): "BUY" or "SELL" for the parent order.
          primary_order_type (str): Order type for the parent order (e.g., "MKT").
          profit_target_price (float or None): Price for the profit taker order (child).
          profit_order_type (str): Order type for the profit order (default "LMT").
          stop_loss_price (float or None): Price for the stop loss order (child).
          stop_order_type (str): Order type for stop loss (default "STP").
          tif (str): Time in force (default "GTC").
          listing_exchange (str): Listing exchange (default "SMART").
          outside_rth (bool): Whether orders are allowed outside regular trading hours.
          referrer (str): Referrer for the parent order.
          is_single_group (bool): If True, adds "isSingleGroup": true to every order.
        """
        base_url = "https://localhost:5000/v1/api/"
        endpoint = f"iserver/account/{self.account_id}/orders"
        
        # Build order identifier exactly as before.
        now = datetime.now()
        c_oid_prefix = ticker.upper() + "_BRACKET_" + now.strftime("%y%m%d%H")
        
        orders = []
        
        # Parent (primary) order
        parent_order = {
            "acctId": self.account_id,
            "conid": int(conid),
            "cOID": c_oid_prefix,
            "orderType": primary_order_type,
            "listingExchange": listing_exchange,
            "outsideRTH": outside_rth,
            "side": "Buy" if primary_side.upper() == "BUY" else "Sell",
            "referrer": referrer,
            "tif": tif,
            "quantity": quantity
        }
        if is_single_group:
            parent_order["isSingleGroup"] = True
        orders.append(parent_order)
        
        # Profit taker order (child order)
        if profit_target_price is not None:
            profit_order = {
                "acctId": self.account_id,
                "conid": int(conid),
                "orderType": profit_order_type,
                "listingExchange": listing_exchange,
                # Setting outsideRTH to false for child orders per sample documentation.
                "outsideRTH": outside_rth,
                "price": profit_target_price,
                "side": "Sell" if primary_side.upper() == "BUY" else "Buy",
                "tif": tif,
                "quantity": quantity,
                "parentId": c_oid_prefix
            }
            if is_single_group:
                profit_order["isSingleGroup"] = True
            orders.append(profit_order)
        
        # Stop loss order (child order)
        if stop_loss_price is not None:
            stop_order = {
                "acctId": self.account_id,
                "conid": int(conid),
                "orderType": stop_order_type,
                "listingExchange": listing_exchange,
                "outsideRTH": outside_rth,
                "price": stop_loss_price,
                "side": "Sell" if primary_side.upper() == "BUY" else "Buy",
                "tif": tif,
                "quantity": quantity,
                "parentId": c_oid_prefix
            }
            if is_single_group:
                stop_order["isSingleGroup"] = True
            orders.append(stop_order)
        
        json_body = {"orders": orders}
        
        # Debug print of the JSON payload
        print("Sending JSON payload:")
        print(json.dumps(json_body, indent=2))
        
        try:
            order_req = requests.post(url=base_url + endpoint, verify=False, json=json_body)
            order_req.raise_for_status()  # Raise exception for 4XX/5XX errors
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                print("Response Content:", e.response.text)
            print(f"An error occurred: {e}")
            return None
    
        order_data = order_req.json()
        print("Status code:", order_req.status_code)
        print("Response Data:", json.dumps(order_data, indent=2))
        
        # Extract and return the order ID from the first element if available.
        if isinstance(order_data, list) and order_data and "id" in order_data[0]:
            return order_data[0]["id"]
        else:
            print("Unexpected response format")
            return None

    
    
    def get_portfolio_summary(self):
        """
        Retrieves the portfolio summary for this account.

        Returns:
            dict or None: Portfolio summary data if available; otherwise, None.
        """
        endpoint = f"portfolio/{self.account_id}/summary"
        request_url = f"{self.base_url}/{endpoint}"
        print("Portfolio Request URL:", request_url)
        response = self.session.get(url=request_url)
        print("HTTP Status Code:", response.status_code)

        try:
            portfolio_data = response.json()
            print("Portfolio Data:", json.dumps(portfolio_data, indent=2))
            return portfolio_data
        except json.JSONDecodeError as e:
            print("Error decoding portfolio JSON:", e)
            print("Raw Response:", response.text)
            return None



    def get_accounts(self):
        """
        Retrieves account details from the IBKR API.
        
        Key Features:
          - Sends a GET request to the "iserver/accounts" endpoint.
          - Returns account information, which may include account numbers,
            balances, and other relevant details.
        
        Returns:
            dict or None: The JSON response containing account details if successful;
                          otherwise, None.
        """
        endpoint = "iserver/accounts"
        request_url = f"{self.base_url}/{endpoint}"
        print("Accounts Request URL:", request_url)
        
        try:
            response = self.session.get(request_url, verify=self.verify_ssl)
            print("HTTP Status Code:", response.status_code)
            if response.status_code == 200:
                accounts_data = response.json()
                print("Accounts Data:")
                print(json.dumps(accounts_data, indent=2))
                return accounts_data
            else:
                print("Failed to retrieve accounts. Status Code:", response.status_code)
                return None
        except Exception as e:
            print("Error retrieving account details:", e)
            return None
    
    

    def get_portfolio_accounts(self):
        """
        Retrieves portfolio account details from the IBKR API.
        
        Process:
          1. Constructs the full URL by combining the base URL with the 'portfolio/accounts' endpoint.
          2. Sends a GET request to this URL.
          3. Parses and prints the JSON response that contains portfolio account details.
        
        Returns:
            dict or None: The JSON data containing portfolio accounts if successful, otherwise None.
        """
        endpoint = "portfolio/accounts"
        request_url = f"{self.base_url}/{endpoint}"
        print("Portfolio Accounts Request URL:", request_url)
        
        try:
            response = self.session.get(url=request_url, verify=self.verify_ssl)
            print("HTTP Status Code:", response.status_code)
            
            data = response.json()
            print("Portfolio Accounts Data:")
            print(json.dumps(data, indent=2))
            return data
        except Exception as e:
            print("Error fetching portfolio accounts:", e)
            return None



    def get_portfolio_summary(self):
        """
        Retrieves the portfolio summary for the account by constructing the URL and sending a GET request.

        The constructed URL is: <base_url>/portfolio/<account_id>/summary

        Returns:
            dict or None: The portfolio summary details if successful; otherwise, None.
        """
        # Construct the endpoint using the account ID.
        endpoint = f"portfolio/{self.account_id}/summary"
        request_url = f"{self.base_url}/{endpoint}"
        print("Portfolio Summary Request URL:", request_url)
        
        try:
            # Send the GET request to the API endpoint.
            response = self.session.get(url=request_url, verify=self.verify_ssl)
            print("HTTP Status Code:", response.status_code)
            
            # Parse and pretty-print the JSON response.
            data = response.json()
            print("Portfolio Summary Data:")
            print(json.dumps(data, indent=2))
            return data
        except Exception as e:
            print("Error fetching portfolio summary:", e)
            return None



    def get_positions(self, direction="a", period="1W", sort="position", model="MyModel"):
        """
        Retrieves positions for the account by constructing the request URL and sending a GET request.
        
        The URL structure is:
            <base_url>/portfolio/<account_id>/positions/0?direction=<direction>&period=<period>&sort=<sort>&model=<model>
        
        Parameters:
            direction (str): Direction of positions; default is "a".
            period (str): Period string for positions; default is "1W".
            sort (str): Sort type for positions; default is "position".
            model (str): Model name; default is "MyModel".
        
        Returns:
            dict or None: The positions data in JSON format if successful; otherwise, None.
        """
        # Build the endpoint using the account id and the literal 'positions/0'
        endpoint = f"portfolio/{self.account_id}/positions/0"
        # Construct the query string with the given parameters.
        query_params = f"?direction={direction}&period={period}&sort={sort}&model={model}"
        # Construct the full request URL.
        request_url = f"{self.base_url}/{endpoint}{query_params}"
        print("Positions Request URL:", request_url)
        
        try:
            # Send the GET request.
            response = self.session.get(url=request_url, verify=self.verify_ssl)
            print("HTTP Status Code:", response.status_code)
            # Parse the JSON response.
            data = response.json()
            print("Positions Data:")
            print(json.dumps(data, indent=2))
            return data
        except Exception as e:
            print("Error fetching positions:", e)
            return None


    
    def receive_market_updates(self, conid):
        ws_url = self.base_url.replace("https://", "wss://") + "/ws"
        
        def on_message(ws, message):
            # Adding an identifier for market updates:
            print("[Market] Received update:")
            print(message)
        
        def on_error(ws, error):
            print("[Market] Error:", error)
        
        def on_close(ws):
            print("[Market] Websocket CLOSED")
        
        def on_open(ws):
            print("[Market] Opened connection")
            time.sleep(3)
            request_message = 'smd+' + conid + '+{"fields":["31","84","86"]}'
            print("[Market] Sending market data request for conid:", conid)
            ws.send(request_message)
        
        ws = websocket.WebSocketApp(
            url=ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
    
    
    def receive_order_updates(self):
        ws_url = self.base_url.replace("https://", "wss://") + "/ws"
        
        def on_message(ws, message):
            # Adding an identifier for order updates:
            print("[Order] Received update:")
            print(message)
        
        def on_error(ws, error):
            print("[Order] Error:", error)
        
        def on_close(ws):
            print("[Order] Websocket CLOSED")
        
        def on_open(ws):
            print("[Order] Opened connection")
            time.sleep(3)
            ws.send('sor+{}')
        
        ws = websocket.WebSocketApp(
            url=ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

