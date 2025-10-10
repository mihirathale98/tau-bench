from tau_bench.types import Action, Task

TASKS = [
    Task(
        annotator="1",
        user_id="mei_thomas_2630",
        instruction="Your user id is mei_thomas_2630. You want the agent to provide detailed information for your reservation with the reservation ID U1YV6I. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'U1YV6I'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="yusuf_martin_3470",
        instruction="Your user id is yusuf_martin_3470. You want to retrieve the details for your existing reservation with the ID BBVDO9. This reservation includes a one-way business class flight from JFK to LAX with a stopover in MIA on May 22, 2024. You are traveling with Yusuf Muller, and your payment history shows you used your Mastercard ending in 6182 for this booking. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'BBVDO9'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="yusuf_patel_4029",
        instruction="Your user id is yusuf_patel_4029. You want to check the details for your one-way flight reservation from Philadelphia to Chicago under reservation ID 847MY1. Please provide all the information related to this booking, including the flight date, flight number, any baggage allowance, and the payment method used. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '847MY1'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="3",
        user_id="raj_garcia_4690",
        instruction="Your user id is raj_garcia_4690. You want to retrieve the details for your reservation with the ID 4NQCM5. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '4NQCM5'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="lucas_rossi_2421",
        instruction="Your user id is lucas_rossi_2421. You want to retrieve the details for your reservation with the id J43KQ8. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'J43KQ8'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="daiki_patel_1917",
        instruction="Your user id is daiki_patel_1917. You want to review the details of your reservation with the ID 7WKBKD. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '7WKBKD'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="ivan_kim_3844",
        instruction="Your user id is ivan_kim_3844. You want to review the details of your existing reservation with the ID Q79V9W. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'Q79V9W'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="yara_silva_8071",
        instruction="Your user id is yara_silva_8071. You want to search for a direct flight from Newark (EWR) to Houston (IAH) on May 18, 2024. Since you have a gold membership, you may inquire about any available benefits that could be applied to this booking. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="search_direct_flight",
                kwargs={'origin': 'EWR', 'destination': 'IAH', 'date': '2024-05-18'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="juan_moore_4540",
        instruction="Your user id is juan_moore_4540. You want to retrieve the details of your reservation with the ID 2P5CYY. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '2P5CYY'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="lei_kim_9517",
        instruction="Your user id is lei_kim_9517. You need to verify your account details, including your current address and saved payment methods. You prefer to use your $100 certificate for any future bookings. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_user_details",
                kwargs={'user_id': 'lei_kim_9517'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="harper_davis_5069",
        instruction="Your user id is harper_davis_5069. You need to retrieve the details for your reservation with the ID WLXS0L. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'WLXS0L'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="isabella_ito_3653",
        instruction="Your user id is isabella_ito_3653. You need details for your flight reservation from Detroit to Minneapolis on May 17. The flight number is HAT210, and your reservation ID is MXCGN8. You are traveling in business class with yourself, Isabella Ito, and another passenger, Aarav Khan. You used your Mastercard ending in 2671 for payment. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'MXCGN8'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="raj_khan_9352",
        instruction="Your user id is raj_khan_9352. You want to verify your account details, including your saved payment methods, upcoming reservations, and any other personal information associated with your account. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_user_details",
                kwargs={'user_id': 'raj_khan_9352'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="liam_lee_5870",
        instruction="Your user id is liam_lee_5870. You want to cancel your existing reservation from Los Angeles to Charlotte with reservation id 1CUG9J. Then, search for a new direct flight from Los Angeles to Charlotte on May 17. Once you find an appropriate flight, book a one-way trip in economy class for Raj Hernandez and Sofia Jackson on May 17, with flight numbers HAT228 and HAT270. You want to use your $49 gift card to pay first, and the remaining balance should be paid with your Mastercard ending in 8261. You also need to include 2 baggages, with 1 being a non-free baggage. Additionally, update your reservation EWO4IQ to change the travel date to May 20 with the same flight numbers HAT299 and HAT047, upgrade to economy class, and add 1 non-free baggage using your Mastercard ending in 8261. You do not want insurance for these bookings. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': '1CUG9J'},
            ),
            Action(
                name="search_direct_flight",
                kwargs={'origin': 'LAX', 'destination': 'CLT', 'date': '2024-05-17'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'liam_lee_5870', 'origin': 'LAX', 'destination': 'CLT', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT228', 'date': '2024-05-17'}, {'flight_number': 'HAT270', 'date': '2024-05-17'}], 'passengers': [{'first_name': 'Raj', 'last_name': 'Hernandez', 'dob': '1964-07-10'}, {'first_name': 'Sofia', 'last_name': 'Jackson', 'dob': '1964-10-10'}], 'payment_methods': [{'payment_id': 'gift_card_6478145', 'amount': 49}, {'payment_id': 'credit_card_1015550', 'amount': 625}], 'total_baggages': 2, 'nonfree_baggages': 1, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'EWO4IQ', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_1015550'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'EWO4IQ', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT299', 'date': '2024-05-20'}, {'flight_number': 'HAT047', 'date': '2024-05-20'}], 'payment_id': 'credit_card_1015550'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="harper_martin_8348",
        instruction="Your user id is harper_martin_8348. You want to make a few changes and bookings for your travel plans. First, please cancel your existing reservation from Detroit to Phoenix with reservation ID MU96D4. Next, you want to book a new one-way flight from New York (JFK) to Miami (MIA) in business class for Mason Rossi, traveling on May 18 on flight number HAT209. You would like to include one checked baggage and opt for insurance. To pay for this, use your gift card of $186 and cover the remaining amount with your Visa card ending in 2492.\n\nAdditionally, you wish to update your existing reservation with ID ER7A5P. Please change the cabin class to business for all flights on this reservation. Also, increase the total baggage allowance to two, including one non-free baggage. Use your Visa card ending in 2492 to cover any additional costs associated with these updates. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="think",
                kwargs={'thought': 'The user harper_martin_8348 is looking to book a new flight and update an existing reservation while efficiently using available payment methods and ensuring all flights have available seats.'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'MU96D4'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'harper_martin_8348', 'origin': 'JFK', 'destination': 'MIA', 'flight_type': 'one_way', 'cabin': 'business', 'flights': [{'flight_number': 'HAT209', 'date': '2024-05-18'}], 'passengers': [{'first_name': 'Mason', 'last_name': 'Rossi', 'dob': '1973-07-25'}], 'payment_methods': [{'payment_id': 'gift_card_3771493', 'amount': 186}, {'payment_id': 'credit_card_4852851', 'amount': 388}], 'total_baggages': 1, 'nonfree_baggages': 1, 'insurance': 'yes'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'ER7A5P', 'total_baggages': 2, 'nonfree_baggages': 1, 'payment_id': 'credit_card_4852851'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'ER7A5P', 'cabin': 'business', 'flights': [{'flight_number': 'HAT165', 'date': '2024-05-26'}, {'flight_number': 'HAT044', 'date': '2024-05-26'}, {'flight_number': 'HAT238', 'date': '2024-05-28'}, {'flight_number': 'HAT105', 'date': '2024-05-28'}], 'payment_id': 'credit_card_4852851'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="3",
        user_id="yusuf_thomas_7802",
        instruction="Your user id is yusuf_thomas_7802. You want to update your existing reservation from Phoenix to San Francisco (reservation id 0SZHSV) by changing the departure flights to May 19 in business class and adding one checked bag. Use your Visa ending in 8264 for the flight changes and your Visa ending in 6833 for the baggage fee. Additionally, you want to cancel your upcoming trip from Phoenix to Los Angeles (reservation id ZI0T78). After canceling, you want to book a new round-trip flight from Phoenix to Los Angeles for Lucas Santos with the following itinerary: depart on May 20 (flight numbers HAT159 and HAT163) and return on May 24 (flight numbers HAT034 and HAT123) in business class. You do not want insurance and prefer to use your $252 gift card first, followed by your $122 gift card, and then your Mastercard ending in 5011 for any remaining balance. You want one bag included in this new booking. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '0SZHSV'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '0SZHSV', 'cabin': 'business', 'flights': [{'flight_number': 'HAT156', 'date': '2024-05-19'}, {'flight_number': 'HAT274', 'date': '2024-05-19'}, {'flight_number': 'HAT204', 'date': '2024-05-28'}, {'flight_number': 'HAT236', 'date': '2024-05-28'}], 'payment_id': 'credit_card_6263035'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': '0SZHSV', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_1321177'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'ZI0T78'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'yusuf_thomas_7802', 'origin': 'PHX', 'destination': 'LAX', 'flight_type': 'round_trip', 'cabin': 'business', 'flights': [{'flight_number': 'HAT159', 'date': '2024-05-20'}, {'flight_number': 'HAT163', 'date': '2024-05-20'}, {'flight_number': 'HAT034', 'date': '2024-05-24'}, {'flight_number': 'HAT123', 'date': '2024-05-24'}], 'passengers': [{'first_name': 'Lucas', 'last_name': 'Santos', 'dob': '1989-07-11'}], 'payment_methods': [{'payment_id': 'gift_card_4714517', 'amount': 252}, {'payment_id': 'gift_card_5627081', 'amount': 122}, {'payment_id': 'credit_card_8008565', 'amount': 920}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="chen_lopez_2451",
        instruction="Your user id is chen_lopez_2451. First, you want to retrieve the details for your reservation with ID 71Y56R. After reviewing, you want to update this reservation to maintain the business class and ensure your flights are as follows: Flight HAT253 on May 18, HAT013 on May 20, HAT161 and HAT045 both on May 28. You prefer to use your Mastercard ending in 2121 for any additional charges. Next, you want to cancel your existing reservation DNL44T. Instead, you wish to book a new one-way trip from Phoenix to Minneapolis on May 18. The flights should be HAT106 followed by HAT254, and you plan to travel in economy class with two passengers, Yara Lopez and Chen Lopez. You will have a total of 2 baggages, of which 1 will incur a fee. Use your Visa ending in 9890 for the payment of this new booking. Finally, you want to update your reservation LP32EB to include a total of 2 baggages, with 1 being a non-free baggage, and pay any related fees with your Mastercard ending in 2121. You do not require insurance for any of these reservations. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '71Y56R'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '71Y56R', 'cabin': 'business', 'flights': [{'flight_number': 'HAT253', 'date': '2024-05-18'}, {'flight_number': 'HAT013', 'date': '2024-05-20'}, {'flight_number': 'HAT161', 'date': '2024-05-28'}, {'flight_number': 'HAT045', 'date': '2024-05-28'}], 'payment_id': 'credit_card_7912636'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'DNL44T'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'chen_lopez_2451', 'origin': 'PHX', 'destination': 'MSP', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT106', 'date': '2024-05-18'}, {'flight_number': 'HAT254', 'date': '2024-05-18'}], 'passengers': [{'first_name': 'Yara', 'last_name': 'Lopez', 'dob': '1992-01-06'}, {'first_name': 'Chen', 'last_name': 'Lopez', 'dob': '1970-02-10'}], 'payment_methods': [{'payment_id': 'credit_card_4073554', 'amount': 574}], 'total_baggages': 2, 'nonfree_baggages': 1, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'LP32EB', 'total_baggages': 2, 'nonfree_baggages': 1, 'payment_id': 'credit_card_7912636'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="amelia_taylor_4937",
        instruction="Your user id is amelia_taylor_4937. You want to make several updates to your reservations. First, for your reservation R8XD2X, you want to change your flights to depart from DFW to SEA on May 19 with flight HAT222, and from SEA to SFO on the same day with flight HAT274. Your return from SFO to SEA on May 22 will be with flight HAT204, and from SEA to DFW with flight HAT037 on the same day. You want to stay in economy class and use your Mastercard ending in 1756 for payment.\n\nNext, for reservation LT77K6, you wish to increase your total baggage count to 5, with 2 being non-free. You prefer to use your $299 gift card for this payment.\n\nAdditionally, you need to cancel your reservation PIMHHE entirely.\n\nYou also want to book a new one-way trip from New York (JFK) to Houston (IAH) on May 18 in business class. You will be the passenger on this reservation, and you want to use your $238 gift card for payment. You do not require any baggage or insurance for this trip.\n\nLastly, for reservation XRC5CB, you want to update the passenger to Emma Kim with the birthdate October 13, 1954. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'R8XD2X', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT222', 'date': '2024-05-19'}, {'flight_number': 'HAT274', 'date': '2024-05-19'}, {'flight_number': 'HAT204', 'date': '2024-05-22'}, {'flight_number': 'HAT037', 'date': '2024-05-22'}], 'payment_id': 'credit_card_1430006'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'LT77K6', 'total_baggages': 5, 'nonfree_baggages': 2, 'payment_id': 'gift_card_4788785'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'PIMHHE'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'amelia_taylor_4937', 'origin': 'JFK', 'destination': 'IAH', 'flight_type': 'one_way', 'cabin': 'business', 'flights': [{'flight_number': 'HAT279', 'date': '2024-05-18'}], 'passengers': [{'first_name': 'Amelia', 'last_name': 'Taylor', 'dob': '1970-01-01'}], 'payment_methods': [{'payment_id': 'gift_card_1822448', 'amount': 209}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_passengers",
                kwargs={'reservation_id': 'XRC5CB', 'passengers': [{'first_name': 'Emma', 'last_name': 'Kim', 'dob': '1954-10-13'}]},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="lucas_taylor_8203",
        instruction="Your user id is lucas_taylor_8203. You want to modify your reservation for the trip from Philadelphia to Houston (reservation id 6BL7WH). You want to upgrade your cabin to business class for all flights on May 19 and May 23, specifically flights HAT291, HAT082, HAT116, and HAT271. Please use your Mastercard ending in 9678 for this upgrade. Additionally, for your one-way trip from Boston to Denver (reservation id U1DEHM), you want to include 3 checked bags, and you prefer to use your gift card with the amount of $292 to cover this. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '6BL7WH', 'cabin': 'business', 'flights': [{'flight_number': 'HAT291', 'date': '2024-05-19'}, {'flight_number': 'HAT082', 'date': '2024-05-19'}, {'flight_number': 'HAT116', 'date': '2024-05-23'}, {'flight_number': 'HAT271', 'date': '2024-05-23'}], 'payment_id': 'credit_card_8476340'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'U1DEHM', 'total_baggages': 3, 'nonfree_baggages': 3, 'payment_id': 'gift_card_2856574'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="raj_muller_5942",
        instruction="Your user id is raj_muller_5942. You want to update your flight reservation with id 41X7CX from New York LaGuardia to Detroit on May 18, changing the cabin to business class on flights HAT245 and HAT265. You want to pay for this change with your Mastercard ending in 7990. Additionally, for your reservation with id 5Q85YP from Boston to Minneapolis, you want to increase your total baggage to 3 bags, including 1 non-free bag. You would like to use your gift card with the balance of $276 to cover any additional charges. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '41X7CX', 'cabin': 'business', 'flights': [{'flight_number': 'HAT245', 'date': '2024-05-18'}, {'flight_number': 'HAT265', 'date': '2024-05-18'}], 'payment_id': 'credit_card_3719965'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': '5Q85YP', 'total_baggages': 3, 'nonfree_baggages': 1, 'payment_id': 'gift_card_2496311'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="3",
        user_id="james_silva_1659",
        instruction="Your user id is james_silva_1659. You want to update your existing reservation (R4H4N6) for a round trip from Minneapolis to New York, ensuring that you fly in economy class. Please adjust the departure flight from Minneapolis to Detroit to May 19 with flight number HAT127, and ensure the Detroit to New York leg is updated to May 20 on flight HAT169. Additionally, modify the return flights: New York to Detroit should be on May 20 with flight number HAT033, and Detroit to Minneapolis on the same day with flight number HAT210. You would also like to add one checked bag to this reservation. For the flight changes, please use your Mastercard ending in 7420 for payment. For the baggage fee, use your gift card. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'R4H4N6', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT127', 'date': '2024-05-19'}, {'flight_number': 'HAT169', 'date': '2024-05-20'}, {'flight_number': 'HAT033', 'date': '2024-05-20'}, {'flight_number': 'HAT210', 'date': '2024-05-20'}], 'payment_id': 'credit_card_1882524'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'R4H4N6', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'gift_card_9230309'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="3",
        user_id="harper_ahmed_9365",
        instruction="Your user id is harper_ahmed_9365. You want to update your reservation with ID 53WBRH to change the flights to an earlier date. You will be flying in business class, and you want the following flights on May 16: HAT180 from IAH to SFO, HAT163 from SFO to LAX, HAT228 from LAX to EWR, and HAT056 from EWR to IAH. To cover this update, you wish to use your gift card amount. Additionally, you want to book a new round-trip reservation from New York (LGA) to Phoenix (PHX), departing on May 16 with flight HAT002 and returning on May 18 with flight HAT051. The passenger for this trip will be Yusuf Johnson, whose details are in your user profile. You prefer to fly in economy class and will have one checked bag. For payment, you want to use your $150 certificate and your $250 certificate, with the remaining balance covered by the latter. You do not require travel insurance. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '53WBRH', 'cabin': 'business', 'flights': [{'flight_number': 'HAT180', 'date': '2024-05-16'}, {'flight_number': 'HAT163', 'date': '2024-05-16'}, {'flight_number': 'HAT228', 'date': '2024-05-16'}, {'flight_number': 'HAT056', 'date': '2024-05-16'}], 'payment_id': 'gift_card_4614903'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'harper_ahmed_9365', 'origin': 'LGA', 'destination': 'PHX', 'flight_type': 'round_trip', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT002', 'date': '2024-05-16'}, {'flight_number': 'HAT051', 'date': '2024-05-18'}], 'passengers': [{'first_name': 'Yusuf', 'last_name': 'Johnson', 'dob': '1959-11-17'}], 'payment_methods': [{'payment_id': 'certificate_1337987', 'amount': 150}, {'payment_id': 'certificate_4335815', 'amount': 222}], 'total_baggages': 1, 'nonfree_baggages': 1, 'insurance': 'no'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="ethan_hernandez_6400",
        instruction="Your user id is ethan_hernandez_6400. For your reservation from Las Vegas to San Francisco (reservation id 7HXRPX), you want to add an additional checked bag, making it a total of 2 bags, with one being a paid bag. You would like to use your Visa card ending in 1332 for this payment. Additionally, for your one-way trip from New York to Boston (reservation id 4069WE), you would like to change your cabin class to economy for both flights on May 19, with flight numbers HAT083 and HAT194. Again, you wish to use the same Visa card ending in 1332 for this transaction. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': '7HXRPX', 'total_baggages': 2, 'nonfree_baggages': 1, 'payment_id': 'credit_card_9038105'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '4069WE', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT083', 'date': '2024-05-19'}, {'flight_number': 'HAT194', 'date': '2024-05-19'}], 'payment_id': 'credit_card_9038105'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="mason_johansson_5154",
        instruction="Your user id is mason_johansson_5154. You want to update your existing reservation for your flight from Charlotte to Denver on May 28 to a new date, May 19, while staying in business class. The flight number is HAT262. You want to pay for this flight change using your Visa card ending in 2961. Additionally, you would like to add one checked bag to this reservation, which will be one non-free baggage, and you prefer to pay for this with your Visa card ending in 1242. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'RB9S17', 'cabin': 'business', 'flights': [{'flight_number': 'HAT262', 'date': '2024-05-19'}], 'payment_id': 'credit_card_5590177'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'RB9S17', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_3358561'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="noah_silva_2256",
        instruction="Your user id is noah_silva_2256. You need to update your existing reservation (ID: T8QHPY) for your flight from Phoenix to Seattle. Please adjust the flight dates to May 19 with flight numbers HAT173 and HAT047, maintaining business class. You want to add one checked baggage to this reservation, using your gift card with the $112 balance for the baggage fee. For the flight adjustments, use your Mastercard ending in 9170. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'T8QHPY', 'cabin': 'business', 'flights': [{'flight_number': 'HAT173', 'date': '2024-05-19'}, {'flight_number': 'HAT047', 'date': '2024-05-19'}], 'payment_id': 'credit_card_7773542'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'T8QHPY', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'gift_card_9130446'},
            ),
        ],
        outputs=['sample_output'],
    ),
    Task(
        annotator="1",
        user_id="mei_lee_8701",
        instruction="Your user id is mei_lee_8701. You need to make two updates. First, for your existing reservation from Dallas (DFW) to Newark (EWR) with reservation id UWNK0D, you want to adjust your return flight. Keep the departure flight as scheduled on May 16 with flight number HAT142, but change the return flight to May 19 using flight number HAT213. Please use your $298 gift card for any additional charges.\n\nSecondly, you want to book a new one-way flight from New York (JFK) to Houston (IAH) on May 19 in economy class. The flight number for this journey is HAT279. The passenger will be Mei Lee, with the birth date of March 5, 1980. You want to include travel insurance for this trip. For payment, apply your $100 certificate first, then your $21 gift card, and finally, cover any remaining balance with your Mastercard ending in 2084. You do not need to add any baggage for this booking. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'UWNK0D', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT142', 'date': '2024-05-16'}, {'flight_number': 'HAT213', 'date': '2024-05-19'}], 'payment_id': 'gift_card_3011170'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'mei_lee_8701', 'origin': 'JFK', 'destination': 'IAH', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT279', 'date': '2024-05-19'}], 'passengers': [{'first_name': 'Mei', 'last_name': 'Lee', 'dob': '1980-03-05'}], 'payment_methods': [{'payment_id': 'certificate_5864879', 'amount': 100}, {'payment_id': 'gift_card_8583604', 'amount': 21}, {'payment_id': 'credit_card_1904381', 'amount': 23}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
        ],
        outputs=['327', '1000', '1786'],
    ),
    Task(
        annotator="1",
        user_id="lucas_rossi_9280",
        instruction="Your user id is lucas_rossi_9280. You need to update your reservation MH743C for your trip from Phoenix to Dallas. You want to change the departure and return dates to May 17 and May 20, respectively, while keeping all flights in business class. The specific flights you want are HAT156 and HAT037 on May 17, and HAT067 and HAT121 on May 20. You want to use your Visa card ending in 1842 for any additional charges related to the flight changes. Additionally, you want to add two checked bags for this reservation and use your gift card valued at $277 to cover the baggage fees. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'MH743C', 'cabin': 'business', 'flights': [{'flight_number': 'HAT156', 'date': '2024-05-17'}, {'flight_number': 'HAT037', 'date': '2024-05-17'}, {'flight_number': 'HAT067', 'date': '2024-05-20'}, {'flight_number': 'HAT121', 'date': '2024-05-20'}], 'payment_id': 'credit_card_7507634'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'MH743C', 'total_baggages': 2, 'nonfree_baggages': 2, 'payment_id': 'gift_card_1600929'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="olivia_jackson_7257",
        instruction="Your user id is olivia_jackson_7257. You want to cancel your existing round-trip reservation from Chicago (ORD) to Phoenix (PHX) with the reservation ID LDZCLM. After canceling, you wish to book a new round-trip flight from Chicago to Phoenix. The flights you prefer are on May 16, with flight numbers HAT230 and HAT035, and returning on May 18 with flight numbers HAT152 and HAT044. You wish to book these flights in business class for two passengers: Aarav Johansson and yourself, Olivia Jackson. You want to pay for this booking using your Visa card ending in 3838. You do not wish to add any baggage or purchase insurance for this reservation. Additionally, for your reservation with ID NQ4Y0O, you would like to update the baggage to a total of 5 bags, of which 2 are non-free, and you want to pay for this update using your Visa card ending in 3838. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'LDZCLM'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'olivia_jackson_7257', 'origin': 'ORD', 'destination': 'PHX', 'flight_type': 'round_trip', 'cabin': 'business', 'flights': [{'flight_number': 'HAT230', 'date': '2024-05-16'}, {'flight_number': 'HAT035', 'date': '2024-05-16'}, {'flight_number': 'HAT152', 'date': '2024-05-18'}, {'flight_number': 'HAT044', 'date': '2024-05-18'}], 'passengers': [{'first_name': 'Aarav', 'last_name': 'Johansson', 'dob': '1983-04-19'}, {'first_name': 'Olivia', 'last_name': 'Jackson', 'dob': '1990-07-25'}], 'payment_methods': [{'payment_id': 'credit_card_2480682', 'amount': 2030}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'NQ4Y0O', 'total_baggages': 5, 'nonfree_baggages': 2, 'payment_id': 'credit_card_2480682'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="sofia_santos_3403",
        instruction="Your user id is sofia_santos_3403. You need to make several updates and bookings: \n\nFirstly, for your existing reservation from Atlanta to Detroit (reservation ID JOHYVS), change the flight dates to May 17, keeping the same flight numbers HAT227 and HAT020, and maintain the business class cabin. Use your gift card to cover any costs.\n\nNext, for your reservation from Houston to Denver (reservation ID 8UW4LT), increase your total baggages to 3, with 2 of them being non-free. Pay for this additional baggage using your gift card.\n\nFinally, you want to book a new one-way flight for Anya Moore from Detroit to Charlotte on May 16, in business class. The flight number is HAT168. This booking should include 1 non-free baggage, and you would like to purchase insurance. Pay for this reservation using your gift card.\n\nYou are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'JOHYVS', 'cabin': 'business', 'flights': [{'flight_number': 'HAT227', 'date': '2024-05-17'}, {'flight_number': 'HAT020', 'date': '2024-05-17'}], 'payment_id': 'gift_card_8467750'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': '8UW4LT', 'total_baggages': 3, 'nonfree_baggages': 2, 'payment_id': 'gift_card_8467750'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'sofia_santos_3403', 'origin': 'DTW', 'destination': 'CLT', 'flight_type': 'one_way', 'cabin': 'business', 'flights': [{'flight_number': 'HAT168', 'date': '2024-05-16'}], 'passengers': [{'first_name': 'Anya', 'last_name': 'Moore', 'dob': '1952-10-26'}], 'payment_methods': [{'payment_id': 'gift_card_8467750', 'amount': 530}], 'total_baggages': 1, 'nonfree_baggages': 1, 'insurance': 'yes'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="yusuf_martin_3470",
        instruction="Your user id is yusuf_martin_3470. You want to cancel your existing reservation from Houston to Phoenix with reservation id UIN4IZ. After that, you want to book a new round-trip flight from Houston (IAH) to Phoenix (PHX) in business class. The flights should be on May 19 with flight numbers HAT180 and HAT123, and return on May 26 with flight number HAT152. The passenger will be Yusuf Muller, with the birthdate of May 6, 1956. You want to include one checked bag and prefer to have travel insurance. For payment, use your $500 certificate first and cover the remaining amount with your Mastercard ending in 6182. Additionally, you want to update your existing reservation FATBVC to include one checked bag, paying for this with your Mastercard ending in 6182. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'UIN4IZ'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'yusuf_martin_3470', 'origin': 'IAH', 'destination': 'PHX', 'flight_type': 'round_trip', 'cabin': 'business', 'flights': [{'flight_number': 'HAT180', 'date': '2024-05-19'}, {'flight_number': 'HAT123', 'date': '2024-05-19'}, {'flight_number': 'HAT152', 'date': '2024-05-26'}], 'passengers': [{'first_name': 'Yusuf', 'last_name': 'Muller', 'dob': '1956-05-06'}], 'payment_methods': [{'payment_id': 'certificate_3071118', 'amount': 500}, {'payment_id': 'credit_card_9067289', 'amount': 741}], 'total_baggages': 1, 'nonfree_baggages': 1, 'insurance': 'yes'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'FATBVC', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_9067289'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="ava_gonzalez_2934",
        instruction="Your user id is ava_gonzalez_2934. You want to update your reservation from San Francisco to LaGuardia (reservation ID HZBXN1) to travel on May 16 instead. You want to fly in business class for all legs of the trip on flights HAT144, HAT226, HAT201, and HAT283. Additionally, you need to update your baggage to include a total of 3 bags, with 2 of them being non-free. For these changes, you prefer to use your Mastercard ending in 6758. Furthermore, you would like to cancel your one-way trip from Newark to Las Vegas (reservation ID 7FVJG2). You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'HZBXN1', 'cabin': 'business', 'flights': [{'flight_number': 'HAT144', 'date': '2024-05-16'}, {'flight_number': 'HAT226', 'date': '2024-05-16'}, {'flight_number': 'HAT201', 'date': '2024-05-16'}, {'flight_number': 'HAT283', 'date': '2024-05-16'}], 'payment_id': 'credit_card_7957134'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'HZBXN1', 'total_baggages': 3, 'nonfree_baggages': 2, 'payment_id': 'credit_card_7957134'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': '7FVJG2'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="juan_sanchez_3680",
        instruction="Your user id is juan_sanchez_3680. You want to update your existing reservation from Charlotte to New York (reservation ID 42IIXI) to fly on May 19, in business class. You will be on flight HAT176 from Charlotte to Detroit and then flight HAT263 from Detroit to New York. You want to pay for this update using your Visa card ending in 8861. Additionally, for your reservation from Philadelphia to Newark (reservation ID PZUNWM), you want to add a total of two bags, with one being chargeable, and you want to pay for this baggage using your $76 gift card. Finally, you wish to book a new one-way flight for Mason Sanchez from Seattle to San Francisco on May 20, in economy class, on flight HAT107. You want to include one checked bag and decline insurance for this booking, using your Visa card ending in 8861 to pay. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': '42IIXI', 'cabin': 'business', 'flights': [{'flight_number': 'HAT176', 'date': '2024-05-19'}, {'flight_number': 'HAT263', 'date': '2024-05-19'}], 'payment_id': 'credit_card_6952762'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'PZUNWM', 'total_baggages': 2, 'nonfree_baggages': 1, 'payment_id': 'gift_card_2850297'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'juan_sanchez_3680', 'origin': 'SEA', 'destination': 'SFO', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT107', 'date': '2024-05-20'}], 'passengers': [{'first_name': 'Mason', 'last_name': 'Sanchez', 'dob': '1992-03-23'}], 'payment_methods': [{'payment_id': 'credit_card_6952762', 'amount': 195}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="mason_johnson_9566",
        instruction="Your user id is mason_johnson_9566. You want to upgrade your Boston to Phoenix round-trip flight (reservation id YWZEQN) to business class while keeping the same flights on May 17, 18, 27, and 28 (flight numbers HAT235, HAT214, HAT181, and HAT217, respectively). You want to add one checked bag to this reservation. Please use your Mastercard ending in 3523 for any additional charges. Additionally, you need to cancel your one-way flight from Seattle to Dallas (reservation id WM6OS0). Lastly, you want to book a new one-way flight from Phoenix to Dallas on May 17 in economy class for yourself, Mason Johnson, born on February 22, 1996, using your Mastercard ending in 3523 to pay the $127 fare. Ensure there is no baggage included in this booking, and you do not require any travel insurance. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'YWZEQN', 'cabin': 'business', 'flights': [{'flight_number': 'HAT235', 'date': '2024-05-17'}, {'flight_number': 'HAT214', 'date': '2024-05-18'}, {'flight_number': 'HAT181', 'date': '2024-05-27'}, {'flight_number': 'HAT217', 'date': '2024-05-28'}], 'payment_id': 'credit_card_3562064'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'YWZEQN', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_3562064'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'WM6OS0'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'mason_johnson_9566', 'origin': 'PHX', 'destination': 'DFW', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT230', 'date': '2024-05-17'}], 'passengers': [{'first_name': 'Mason', 'last_name': 'Johnson', 'dob': '1996-02-22'}], 'payment_methods': [{'payment_id': 'credit_card_3562064', 'amount': 127}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
        ],
        outputs=['327', '1000', '1786'],
    ),
    Task(
        annotator="3",
        user_id="anya_lee_9572",
        instruction="Your user id is anya_lee_9572. You want to cancel your existing reservation from JFK to LAS, which has the reservation ID 7KYHMW. After that, you wish to rebook a round-trip flight from New York (JFK) to Las Vegas (LAS) departing on May 20 and returning on May 28. You prefer to travel in economy class on the following flights: HAT136 and HAT281 on May 20, and HAT047 and HAT021 on May 28. You will be traveling with one piece of baggage and would like to include insurance. You want to use your Visa card ending in 3963 to pay for this booking. Additionally, you wish to update your existing reservation with ID ABB0M7 to confirm that all flights on May 28 and 30 remain in economy class, and charge any necessary fees to your Visa card ending in 9975. Lastly, please update your reservation ID I6KKNF to include a total of three pieces of baggage, with one being a non-free baggage, charged to your Mastercard ending in 1507. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': '7KYHMW'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'anya_lee_9572', 'origin': 'JFK', 'destination': 'LAS', 'flight_type': 'round_trip', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT136', 'date': '2024-05-20'}, {'flight_number': 'HAT281', 'date': '2024-05-20'}, {'flight_number': 'HAT047', 'date': '2024-05-28'}, {'flight_number': 'HAT021', 'date': '2024-05-28'}], 'passengers': [{'first_name': 'Anya', 'last_name': 'Lee', 'dob': '1986-07-25'}], 'payment_methods': [{'payment_id': 'credit_card_4589036', 'amount': 665}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'ABB0M7', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT011', 'date': '2024-05-28'}, {'flight_number': 'HAT163', 'date': '2024-05-28'}, {'flight_number': 'HAT186', 'date': '2024-05-30'}, {'flight_number': 'HAT099', 'date': '2024-05-30'}], 'payment_id': 'credit_card_4390028'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'I6KKNF', 'total_baggages': 3, 'nonfree_baggages': 1, 'payment_id': 'credit_card_9909970'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="mia_silva_9133",
        instruction="Your user id is mia_silva_9133. You want to update your reservation with ID YZF0F6. Change the cabin class to business for the flight from Houston to Atlanta, with flight numbers HAT085 on May 19 and HAT218 on May 20. Use your Visa card ending in 2436 for this payment. Additionally, for the reservation ID TD3FPM, add one checked bag and use your $234 gift card for payment. You also want to cancel your reservation ID P1D9KS entirely. Finally, book a new round-trip flight from Dallas to Los Angeles in business class for Amelia Johansson. The flights should be on May 18 (HAT124) and May 20 (HAT022). Include one checked bag and insurance for this trip. Pay with your $203 gift card and the remaining balance with your Visa card ending in 7854. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'YZF0F6', 'cabin': 'business', 'flights': [{'flight_number': 'HAT085', 'date': '2024-05-19'}, {'flight_number': 'HAT218', 'date': '2024-05-20'}], 'payment_id': 'credit_card_9663703'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'TD3FPM', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'gift_card_1267960'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'P1D9KS'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'mia_silva_9133', 'origin': 'DFW', 'destination': 'LAX', 'flight_type': 'round_trip', 'cabin': 'business', 'flights': [{'flight_number': 'HAT124', 'date': '2024-05-18'}, {'flight_number': 'HAT022', 'date': '2024-05-20'}], 'passengers': [{'first_name': 'Amelia', 'last_name': 'Johansson', 'dob': '1966-05-22'}], 'payment_methods': [{'payment_id': 'gift_card_5086914', 'amount': 203}, {'payment_id': 'credit_card_3163658', 'amount': 518}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="harper_davis_5069",
        instruction="Your user id is harper_davis_5069. You need to retrieve the details for your reservation with the ID WLXS0L. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'WLXS0L'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="liam_muller_3384",
        instruction="Your user id is liam_muller_3384. You want to retrieve the details of your reservation with the ID HDUF3Q. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'HDUF3Q'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="harper_santos_6381",
        instruction="Your user id is harper_santos_6381. You want to inquire about the details of your existing reservation with the reservation ID IER616. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'IER616'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="lucas_wilson_8118",
        instruction="Your user id is lucas_wilson_8118. You want to retrieve the details of your reservation with the reservation ID I6XC2H. You prefer not to provide any additional information unless specifically asked by the agent. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'I6XC2H'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="amelia_hernandez_8403",
        instruction="Your user id is amelia_hernandez_8403. You want to retrieve the details for your reservation with the ID 5JR4XX. Please ensure you check all aspects of the booking, including the flight itinerary, passenger information, and payment details. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': '5JR4XX'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="mason_johansson_5154",
        instruction="Your user id is mason_johansson_5154. You want to retrieve the details of your reservation with the ID RB9S17. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'RB9S17'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="mason_johansson_5154",
        instruction="Your user id is mason_johansson_5154. You want to retrieve the details of your reservation with the ID RB9S17. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'RB9S17'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="juan_li_9671",
        instruction="Your user id is juan_li_9671. You want to cancel your current reservation from Charlotte to Orlando under reservation ID PPHW67. After that, you are searching for direct flights from Charlotte to Orlando on May 17, 2024. Once you find a suitable option, you wish to book a round-trip in economy class for Daiki Garcia and Lei Garcia, with flights HAT064 on May 17 and HAT017 on May 19. You want 2 checked bags and prefer to include insurance. For payment, use your $500 certificate first, and then the remaining balance with your Visa ending in 8442.\n\nAdditionally, you want to upgrade your existing reservation Y2DJ0A to business class while keeping the same flights on May 25, 2024, and pay for any differences using your Visa ending in 8442. Lastly, for reservation ITSLB7, you want to update the number of total baggages to 4, with 2 of them being non-free, and pay for this update using your gift card worth $120. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'PPHW67'},
            ),
            Action(
                name="search_direct_flight",
                kwargs={'origin': 'CLT', 'destination': 'MCO', 'date': '2024-05-17'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'juan_li_9671', 'origin': 'CLT', 'destination': 'MCO', 'flight_type': 'round_trip', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT064', 'date': '2024-05-17'}, {'flight_number': 'HAT017', 'date': '2024-05-19'}], 'passengers': [{'first_name': 'Daiki', 'last_name': 'Garcia', 'dob': '1973-07-19'}, {'first_name': 'Lei', 'last_name': 'Garcia', 'dob': '1962-08-08'}], 'payment_methods': [{'payment_id': 'certificate_4380964', 'amount': 500}, {'payment_id': 'credit_card_3086580', 'amount': 146}], 'total_baggages': 2, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'Y2DJ0A', 'cabin': 'business', 'flights': [{'flight_number': 'HAT270', 'date': '2024-05-25'}, {'flight_number': 'HAT064', 'date': '2024-05-25'}], 'payment_id': 'credit_card_3086580'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'ITSLB7', 'total_baggages': 4, 'nonfree_baggages': 2, 'payment_id': 'gift_card_7745140'},
            ),
        ],
        outputs=['sample_output'],
    ),
    Task(
        annotator="0",
        user_id="ivan_garcia_1794",
        instruction="Your user id is ivan_garcia_1794. You want to update your existing reservation LV5MG2 from Las Vegas to Chicago, flying on May 20, to business class. You prefer to keep the same flights, HAT115 and HAT129, and you want to add two checked baggages. Please use your Visa card ending in 8790 for any additional charges. Additionally, you want to book a new one-way flight from Atlanta to Dallas on May 19 in economy class with flight number HAT059. This reservation will be for yourself, Ivan Garcia, and you want to use your Visa card ending in 8149 for payment. Include travel insurance and one carry-on bag for this new booking. Lastly, you wish to cancel your reservation PG7O11. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'LV5MG2'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'LV5MG2', 'cabin': 'business', 'flights': [{'flight_number': 'HAT115', 'date': '2024-05-20'}, {'flight_number': 'HAT129', 'date': '2024-05-20'}], 'payment_id': 'credit_card_8638712'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'LV5MG2', 'total_baggages': 2, 'nonfree_baggages': 2, 'payment_id': 'credit_card_8638712'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'ivan_garcia_1794', 'origin': 'ATL', 'destination': 'DFW', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT059', 'date': '2024-05-19'}], 'passengers': [{'first_name': 'Ivan', 'last_name': 'Garcia', 'dob': '1980-02-15'}], 'payment_methods': [{'payment_id': 'credit_card_7155120', 'amount': 137}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': 'PG7O11'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="mei_patel_4436",
        instruction="Your user id is mei_patel_4436. You want to update your reservation with ID E17VRA to change the cabin class from basic economy to economy for your flights from LaGuardia to Seattle, which include flight HAT114 on May 19 and flight HAT156 on May 20. You prefer to use your Mastercard ending in 1562 for this upgrade. Additionally, you want to add 3 checked bags to this reservation, and for this, you wish to use your Mastercard ending in 4094. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'E17VRA', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT114', 'date': '2024-05-19'}, {'flight_number': 'HAT156', 'date': '2024-05-20'}], 'payment_id': 'credit_card_2126547'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'E17VRA', 'total_baggages': 3, 'nonfree_baggages': 3, 'payment_id': 'credit_card_4435842'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="0",
        user_id="liam_lee_5870",
        instruction="Your user id is liam_lee_5870. You want to cancel your existing reservation from Los Angeles to Charlotte with reservation id 1CUG9J. Then, search for a new direct flight from Los Angeles to Charlotte on May 17. Once you find an appropriate flight, book a one-way trip in economy class for Raj Hernandez and Sofia Jackson on May 17, with flight numbers HAT228 and HAT270. You want to use your $49 gift card to pay first, and the remaining balance should be paid with your Mastercard ending in 8261. You also need to include 2 baggages, with 1 being a non-free baggage. Additionally, update your reservation EWO4IQ to change the travel date to May 20 with the same flight numbers HAT299 and HAT047, upgrade to economy class, and add 1 non-free baggage using your Mastercard ending in 8261. You do not want insurance for these bookings. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': '1CUG9J'},
            ),
            Action(
                name="search_direct_flight",
                kwargs={'origin': 'LAX', 'destination': 'CLT', 'date': '2024-05-17'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'liam_lee_5870', 'origin': 'LAX', 'destination': 'CLT', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT228', 'date': '2024-05-17'}, {'flight_number': 'HAT270', 'date': '2024-05-17'}], 'passengers': [{'first_name': 'Raj', 'last_name': 'Hernandez', 'dob': '1964-07-10'}, {'first_name': 'Sofia', 'last_name': 'Jackson', 'dob': '1964-10-10'}], 'payment_methods': [{'payment_id': 'gift_card_6478145', 'amount': 49}, {'payment_id': 'credit_card_1015550', 'amount': 625}], 'total_baggages': 2, 'nonfree_baggages': 1, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'EWO4IQ', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'credit_card_1015550'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'EWO4IQ', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT299', 'date': '2024-05-20'}, {'flight_number': 'HAT047', 'date': '2024-05-20'}], 'payment_id': 'credit_card_1015550'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="daiki_martin_9991",
        instruction="Your user id is daiki_martin_9991. You want to update your existing reservation from Seattle to New York (reservation ID MH0T63) to change the departure date to May 18, while retaining the same flights: HAT121, HAT081, HAT245, and HAT156. You prefer to fly in economy class for this trip. You want to use your Mastercard ending in 5570 for any additional charges related to this update. Additionally, you would like to add one checked baggage to this reservation, and you prefer to use your $194 gift card for this baggage fee. \n\nSeparately, you want to book a one-way flight for Ethan Moore from Phoenix to Seattle on May 20, with flight number HAT156. You prefer the basic economy cabin for this flight. You plan to use your $194 gift card to cover the cost of this booking. You do not want to purchase travel insurance. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'MH0T63', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT121', 'date': '2024-05-18'}, {'flight_number': 'HAT081', 'date': '2024-05-18'}, {'flight_number': 'HAT245', 'date': '2024-05-18'}, {'flight_number': 'HAT156', 'date': '2024-05-18'}], 'payment_id': 'credit_card_6921949'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'MH0T63', 'total_baggages': 1, 'nonfree_baggages': 1, 'payment_id': 'gift_card_9217496'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'daiki_martin_9991', 'origin': 'PHX', 'destination': 'SEA', 'flight_type': 'one_way', 'cabin': 'basic_economy', 'flights': [{'flight_number': 'HAT156', 'date': '2024-05-20'}], 'passengers': [{'first_name': 'Ethan', 'last_name': 'Moore', 'dob': '1962-05-19'}], 'payment_methods': [{'payment_id': 'gift_card_9217496', 'amount': 80}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="1",
        user_id="ivan_muller_7015",
        instruction="Your user id is ivan_muller_7015. You want to upgrade your existing reservation (G72NSF) for your journey from Detroit to Seattle on May 17. You prefer to change your cabin class from economy to business on both flights (HAT097 from Detroit to Phoenix and HAT251 from Phoenix to Seattle). You will use your Mastercard ending in 6710 for this upgrade. Additionally, you want to increase your total checked baggage from 2 to 3, with one non-free baggage. You want to use your gift card worth $128 to pay for the baggage fees. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'G72NSF', 'cabin': 'business', 'flights': [{'flight_number': 'HAT097', 'date': '2024-05-17'}, {'flight_number': 'HAT251', 'date': '2024-05-17'}], 'payment_id': 'credit_card_3563913'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'G72NSF', 'total_baggages': 3, 'nonfree_baggages': 1, 'payment_id': 'gift_card_8516878'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="mia_lopez_6592",
        instruction="Your user id is mia_lopez_6592. You need details for your reservation with ID P9YQCF. You want to update the flights for this reservation to fly on May 19, changing to flight numbers HAT235 and HAT153, and on May 20, to flight numbers HAT032 and HAT026, all in business class. You want to pay for this update with your Visa ending in 3305. Additionally, you want to increase your total baggage allowance to 6 bags, with 1 being a non-free bag, using your $135 gift card for payment. Furthermore, you want to book a one-way business class flight from Denver to Atlanta on May 19 for Noah Silva, with flight number HAT261. You prefer to pay using your $250 certificate first and cover any remaining balance with your Visa ending in 3305. You want insurance for this booking. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="get_reservation_details",
                kwargs={'reservation_id': 'P9YQCF'},
            ),
            Action(
                name="update_reservation_flights",
                kwargs={'reservation_id': 'P9YQCF', 'cabin': 'business', 'flights': [{'flight_number': 'HAT235', 'date': '2024-05-19'}, {'flight_number': 'HAT153', 'date': '2024-05-19'}, {'flight_number': 'HAT032', 'date': '2024-05-20'}, {'flight_number': 'HAT026', 'date': '2024-05-20'}], 'payment_id': 'credit_card_9314282'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'P9YQCF', 'total_baggages': 6, 'nonfree_baggages': 1, 'payment_id': 'gift_card_3319320'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'mia_lopez_6592', 'origin': 'DEN', 'destination': 'ATL', 'flight_type': 'one_way', 'cabin': 'business', 'flights': [{'flight_number': 'HAT261', 'date': '2024-05-19'}], 'passengers': [{'first_name': 'Noah', 'last_name': 'Silva', 'dob': '1995-01-20'}], 'payment_methods': [{'payment_id': 'certificate_5677938', 'amount': 250}, {'payment_id': 'credit_card_9314282', 'amount': 148}], 'total_baggages': 1, 'nonfree_baggages': 0, 'insurance': 'yes'},
            ),
        ],
        outputs=[],
    ),
    Task(
        annotator="2",
        user_id="ivan_taylor_6615",
        instruction="Your user id is ivan_taylor_6615. You need to cancel your current one-way flight reservation from Boston to Charlotte on May 12 under reservation ID 7GJ1NY for Aarav Kim. After canceling, you want to book a new one-way flight from Atlanta to Charlotte on May 19 in economy class for Aarav Kim. Use your $500 certificate to cover the cost of this new reservation. Additionally, for your round-trip reservation from Los Angeles to Houston with reservation ID PK9XO8, you want to update the baggage allowance to include 3 total bags, with 1 being a non-free checked bag. Use your Visa card ending in 1656 for any additional charges related to the baggage update. You do not want to add any insurance. You are reactive to the agent and will not say anything that is not asked.",
        actions=[
            Action(
                name="cancel_reservation",
                kwargs={'reservation_id': '7GJ1NY'},
            ),
            Action(
                name="book_reservation",
                kwargs={'user_id': 'ivan_taylor_6615', 'origin': 'ATL', 'destination': 'CLT', 'flight_type': 'one_way', 'cabin': 'economy', 'flights': [{'flight_number': 'HAT260', 'date': '2024-05-19'}], 'passengers': [{'first_name': 'Aarav', 'last_name': 'Kim', 'dob': '1962-10-28'}], 'payment_methods': [{'payment_id': 'certificate_1960821', 'amount': 196}], 'total_baggages': 0, 'nonfree_baggages': 0, 'insurance': 'no'},
            ),
            Action(
                name="update_reservation_baggages",
                kwargs={'reservation_id': 'PK9XO8', 'total_baggages': 3, 'nonfree_baggages': 1, 'payment_id': 'credit_card_1885633'},
            ),
        ],
        outputs=[],
    ),
]
