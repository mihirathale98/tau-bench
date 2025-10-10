from tau_bench.types import Action, Task

TASKS = [
    Task(
        annotator="1",
        user_id="mei_thomas_2630",
        instruction="Your user id is mei_thomas_2630. You want to retrieve the details of a specific reservation, though you do not recall the reservation id. You prefer the agent to ask for necessary information, as you are reactive and will not provide details unless prompted. Your birthday is in your user profile, so you do not prefer to provide it. If and only if additional identification is required, you are open to verifying your identity with your stored personal details.",
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
        instruction="Your user id is yusuf_martin_3470. You want to retrieve the details of your existing reservation. You do not have the reservation id memorized, so you depend on the agent to assist you in finding it. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If additional information is needed, you prefer the agent guides you through the process.",
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
        instruction="Your user id is yusuf_patel_4029. You want to inquire about the details of an existing reservation with the reservation id you cannot recall right now. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If the agent asks for additional information to find your reservation, you are willing to provide it if available.",
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
        instruction="Your user id is raj_garcia_4690. You want to retrieve the details of your existing reservation but cannot recall the reservation id. If and only if the agent needs further information, you are willing to confirm the trip dates or destinations. However, you prefer not to provide your birthday because it is already in your user profile. You are reactive to the agent and will not say anything that is not asked. You tend to be concise and direct in your communication, ensuring efficiency and clarity in your interactions.",
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
        instruction="Your user id is lucas_rossi_2421. You need to review the details of an existing reservation, but you do not remember the reservation id. If the agent can find it, you want to confirm the flight details and ensure everything is in order. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. Please handle any sensitive information carefully, as you value your privacy.",
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
        instruction="Your user id is daiki_patel_1917. You want to retrieve the details of your reservation but do not remember the reservation id exactly; however, the agent might find it under a similar reference. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If and only if the agent cannot find the reservation, you are willing to provide more information if necessary.",
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
        instruction="Your user id is ivan_kim_3844. You want to retrieve the details of your existing reservation. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked. If and only if the reservation details are not retrievable, you would be open to discussing alternative options with the agent.",
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
        instruction="Your user id is yara_silva_8071. You want to find a direct flight from Newark to Houston on May 18. You prefer a straightforward journey without stopovers. If and only if there are no direct flights available, you are open to considering options with one stopover. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is juan_moore_4540. You want to inquire about the details of your reservation, but you do not recall the reservation id. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If and only if asked, you may confirm that you are interested in ensuring all details are correct and up-to-date, but you will rely on the agent to guide the conversation.",
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
        instruction="Your user id is lei_kim_9517. You are planning a trip and require the agent to first confirm your user details. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. You want to ensure all personal details are correctly captured in your profile before proceeding with any bookings or inquiries.",
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
        instruction="Your user id is harper_davis_5069. You want to retrieve the details for a specific reservation but cannot remember the reservation id. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If and only if asked to verify your identity, you prefer to provide information available in your user profile.",
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
        instruction="Your user id is isabella_ito_3653. You need the details for your current reservation but do not have the reservation id readily available. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked. If and only if further confirmation is needed, you are willing to answer specific questions about the reservation details.",
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
        instruction="Your user id is raj_khan_9352. You are seeking assistance from the agent to retrieve your user details. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. You are patient and await further questions or instructions from the agent to proceed with any additional requests.",
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
        instruction="Your user id is liam_lee_5870. You are looking to cancel your current reservation for reasons not specified here. You need to find and book a direct flight from Los Angeles to Charlotte on May 17, traveling one way in economy class. You prefer to travel with Raj Hernandez and Sofia Jackson, with their birthdays already in the profile, so you do not prefer to provide them. You have 2 bags, and one of them is a non-free checked bag. You do not want to purchase insurance. You wish to pay using your gift card first, and if it doesn’t cover the full amount, use your 1015550 card for the remaining balance. Additionally, you need to update your existing reservation to include 1 non-free checked bag and change the flights to the ones on May 20. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is harper_martin_8348. You want to cancel your current reservation with the id MU96D4. Subsequently, you are looking to book a new business class one-way flight from New York to Miami on May 18 with the passenger Mason Rossi, born on July 25, 1973. You prefer to use your gift card first for payment, covering the amount of $186, and then use your 4852 card for the remaining $388. You have a total of 1 baggage, which is non-free, and you would like to include insurance in this booking. Additionally, you need to update an existing reservation, id ER7A5P, to include 2 baggages (1 non-free) and ensure all flights are in business class, paying with your 4852 card. Your birthday is in your user profile so you do not prefer to provide it. If and only if there are issues with seat availability, you would like to be informed to adjust your plans. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is yusuf_thomas_7802. You want to first get the details of your current reservation for clarity. You are looking to update this reservation to include flights on May 19 and May 28 in business class. You have one bag to check, and you are okay with using your 6263 card to pay for the flight updates and your 1321 card for the baggage fees. If there is any issue with these cards, you want the agent to ask for alternatives. You also need to cancel a different reservation, which you can identify by its reservation ID. Following that, you plan to book a new round-trip flight from Phoenix to Los Angeles, leaving on May 20 and returning on May 24, traveling in business class. You will be flying with one passenger, Lucas Santos, whose details are already in the system. You prefer to pay using your gift cards first, using your credit card for any remaining balance. If and only if the gift cards cannot cover the cost, you will use your 8008 card to complete the payment. You have just one non-free bag to check on this new trip and do not require any travel insurance. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is chen_lopez_2451. You need to retrieve the details of your existing reservation for an upcoming trip. You want to update this reservation to travel in business class, utilizing your 7912636 card for payment. If and only if this reservation cannot be updated, you would like to cancel reservation DNL44T. You also want to book a new one-way flight from Phoenix to Minneapolis on May 18 for two passengers, Yara Lopez and yourself, Chen Lopez. The trip should be in economy class, and you have a preference for using your 4073554 card for the payment. You will have two pieces of luggage, one of which requires additional payment. You prefer not to purchase insurance. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is amelia_taylor_4937. You want to update your existing reservation to include a round-trip flight from JFK to IAH with the outbound on May 19 and return on May 22, all in economy class. You also need to add 5 total baggages, with 2 being nonfree. You prefer to pay for the flights using your 7447 card and for the baggage with your gift card that covers it entirely. Additionally, you want to cancel another reservation you have without specifying the details. You also wish to book a new one-way flight from JFK to IAH on May 18 in business class for yourself, and you want to use your gift card for the full payment. You do not require any baggage for this new booking, nor do you want insurance. You wish to update another reservation to change the passenger to Emma Kim, born on October 13, 1954. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is lucas_taylor_8203. You want to update your reservation for a round trip, ensuring that you're flying in business class. Your flights are scheduled to depart on May 19 and return on May 23. You are not concerned about stopovers and prefer the itinerary you already have. For the flights, you want to use your 8476 card for payment. Additionally, you need to update the baggage allowance on a separate reservation, requiring 3 checked bags, all of which are non-free, and you prefer to pay for this with a gift card. If and only if the gift card cannot cover the full amount, you would allow using your credit card for the remaining balance. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is raj_muller_5942. You want to update your reservation to include business class flights for your trip on May 18. You are flexible with flight times but prefer to have minimal layovers. You plan to bring 3 pieces of luggage, with one of them requiring payment. You wish to use your 7447 card to pay for the flight update, and prefer using your gift card for the baggage charges. If and only if the gift card cannot cover the baggage fees, you will use your 7447 card instead. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is james_silva_1659. You want to update your existing reservation for a multi-leg trip starting on May 19. You prefer to fly in economy class and need the flights to be updated to include all legs on May 19 and May 20. You require one checked bag, and if it incurs an additional fee, you want to use your gift card to cover this cost. For the flight updates, you prefer to pay with your 1882 card. If and only if the gift card cannot cover the baggage fee, use the 1882 card for the remaining balance. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is harper_ahmed_9365. You want to update your existing reservation (reservation id not remembered) for a flight on May 16, upgrading to business class. You will be flying across multiple flights on that day and prefer to use your gift card to pay for these changes. Additionally, you want to book a new round-trip flight from New York LaGuardia to Phoenix, departing on May 16 and returning on May 18, flying in economy class. You will be booking this trip for Yusuf Johnson. You have one piece of non-free baggage for this new reservation and do not require insurance. You wish to use your two certificates to cover the payment for this booking. If only one certificate can be used, prioritize the one with the larger balance. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is ethan_hernandez_6400. You want to update your reservations to include two baggages, with one being non-free, for your upcoming travel. You prefer to pay with your 9038 card. Additionally, you wish to ensure your flights are in economy class for the trip occurring on May 19. If and only if there are any issues with the flight changes, you are open to discussing alternatives on the same day. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is mason_johansson_5154. You want to update your reservation for a flight on May 19, upgrading to business class. You want to ensure you have one checked bag for which you need to pay. Use your 5590 card for the flight upgrade and your 3358 card for the baggage fees. If and only if there is an issue with using your 5590 card, you prefer to use the 3358 card for the flight as well. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. It's essential that your travel arrangements are handled smoothly and efficiently.",
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
        instruction="Your user id is noah_silva_2256. You want to update your existing reservation for a trip, ensuring you travel in business class on May 19. You are looking for flights that possibly include the most comfortable options available. You have a preference for using your 7773 card for flight payments. Additionally, you need to check in one bag, which will incur an extra charge, and you prefer to pay for this baggage using your gift card. If and only if your preferred payment methods cannot cover the costs, you authorize the agent to suggest alternative options. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is mei_lee_8701. You want to update your existing reservation to include flights on May 16 and May 19, staying in economy class. You prefer to pay using your gift card with the balance that fits. You also need to book a one-way flight from New York to Houston on May 19 in economy class. You have no baggage and you require insurance. You prefer to use your certificate first for payment, followed by your gift card, and finally your 1904 card for any remaining balance. If and only if the certificate cannot be used fully, you would like to maximize its application before using the other methods. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is lucas_rossi_9280. You want to update your reservation to fly in business class for your upcoming trip, ensuring the flights are on May 17 and May 20. You require 2 pieces of checked luggage and prefer to cover these fees with a gift card. If the cost of the flight updates cannot be fully covered by the gift card, use your 7507634 card to pay for the flights. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is olivia_jackson_7257. You need to cancel your current reservation, though you do not remember the reservation ID. You want to book a round-trip flight from Chicago to Phoenix, departing on May 16 and returning on May 18. You prefer to travel in business class. You will be traveling with two passengers, Aarav Johansson and yourself, Olivia Jackson. You do not need to provide your birthday as it is already in your user profile. You prefer to pay using your 2480 card for the total amount of $2030. You do not want any checked bags for this trip, and you prefer not to purchase travel insurance. Additionally, you need to update a separate reservation to include 5 total bags, of which 2 are non-free, using your 2480 card for payment. If and only if the agent needs more information, you are reactive and will only provide answers to their questions.",
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
        instruction="Your user id is sofia_santos_3403. You want to update your reservation for a flight on May 17, ensuring you are flying in business class. You wish to use your gift card for payment. Additionally, you need to update another reservation to include 3 baggages, 2 of which are not free, paid with the same gift card. You also want to book a one-way flight from Detroit to Charlotte on May 16 for Anya Moore, traveling in business class with 1 checked baggage and insurance. You prefer to pay with your gift card. If the gift card cannot cover all expenses, you do not provide any other payment details unless asked. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is yusuf_martin_3470. You want to cancel your current reservation with the id UIN4IZ and book a new round-trip flight from Houston to Phoenix, departing on May 19 and returning on May 26. You prefer to travel in business class. You want to bring one checked bag, and you are open to having a non-direct flight option. You wish to purchase travel insurance. You prefer to use your certificate for payment; if and only if there is a remaining balance, you would like to use your 9067 card for the rest. You have updated your reservation with id FATBVC to reflect one checked bag and want to pay for this with your 9067 card. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is ava_gonzalez_2934. You want to update your reservation for an upcoming trip on May 16 to include traveling in business class with flights that provide the best schedule, even if it involves multiple connections. You wish to travel with a total of 3 checked bags, of which 2 will incur additional fees, and you prefer to use your 7957 card for payment. If and only if it is necessary to make any changes, you prefer to cancel your reservation 7FVJG2 completely. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is juan_sanchez_3680. You want to update your existing reservation to fly in business class on May 19 and ensure your flights are confirmed for the quickest journey on that date. You prefer to use your 6952 card for payment. Additionally, you need to update another reservation to include 2 total baggages, with 1 being non-free, and prefer using your gift card for this. Furthermore, you want to book a one-way flight from Seattle to San Francisco on May 20 for Mason Sanchez, traveling in economy. You do not require insurance for this booking and wish to use your 6952 card to pay the full amount. Your birthday is in your user profile so you do not prefer to provide it. If and only if payment with your 6952 card is unsuccessful, you will consider other payment options upon the agent's prompt. You are reactive to the agent and will not say anything that is not asked. You are also eager to finalize these arrangements promptly.",
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
        instruction="Your user id is mason_johnson_9566. You want to update your flights for an existing reservation to business class and ensure it includes the following legs: on May 17 and May 18, as well as return legs on May 27 and May 28. You prefer to use your 3562 card for payment. Additionally, you need to update your baggage for this reservation to include one checked bag. You also need to cancel another reservation that you no longer require. For a new booking, you want to fly one way from Phoenix to Dallas on May 17 in economy class, with no baggage or insurance. Payment for this new booking should also be made using your 3562 card. If and only if this card cannot be used, you will provide another method when asked. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is anya_lee_9572. You want to cancel your existing reservation and instead book a round-trip flight from New York to Las Vegas, departing on May 20 and returning on May 28, in economy class. You will be traveling with one baggage, and you prefer to have travel insurance. You wish to pay using your 4589 card. If there are multiple flights available on the same day, you prefer the one with the shortest total travel time. You also need to update a separate reservation for another trip on May 28 and May 30, ensuring it remains in economy class and using your 4390 card for payment. Additionally, you want to modify a different reservation to include a total of three baggages, one of which is nonfree, and prefer to pay with your 9909 card. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is mia_silva_9133. You want to modify your travel plans for an upcoming trip. First, update your flight reservations to business class and change to flights departing on May 19 and returning on May 20. You prefer to pay for this with your 9663703 card. Additionally, you need to adjust your baggage reservation to include one checked bag, for which you will use your gift card. You also want to cancel a separate reservation that you no longer need. Furthermore, you want to book a new round-trip ticket from Dallas to Los Angeles, departing on May 18 and returning on May 20, in business class. You will be traveling under the name Amelia Johansson. Your payment preference is to use your gift card of 203, and if there's any remaining balance, you will cover it with your 3163658 card. You need one checked bag, but do not require additional paid baggage. You want to include travel insurance for this new booking. If and only if the situation requires clarifications, you will react to the agent's questions, but you will not volunteer additional information unprompted. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is harper_davis_5069. You want to retrieve the details for a specific reservation but cannot remember the reservation id. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If and only if asked to verify your identity, you prefer to provide information available in your user profile.",
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
        instruction="Your user id is liam_muller_3384. You want to inquire about the details of your current reservation, but you don't recall the reservation id. If and only if the details cannot be retrieved with the information available, you are open to providing additional information to assist the agent in locating your reservation. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is harper_santos_6381. You want to retrieve the details of your reservation, which you have on file but do not remember the specifics. Your birthday is in your user profile, so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked. You would appreciate any assistance the agent can provide in quickly retrieving this information.",
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
        instruction="Your user id is lucas_wilson_8118. You need to retrieve the details for a reservation, but you don't remember the specifics except for the reservation id, which is 'I6XC2H'. You do not prefer to provide your birthday as it is already in your user profile. If and only if the agent asks for additional information, you will provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is amelia_hernandez_8403. You want to retrieve the details of your reservation, but you do not recall your reservation id. If and only if the agent asks, you can provide additional information to help locate your booking. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If payment information is needed, you are open to discussing your options with the agent.",
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
        instruction="Your user id is mason_johansson_5154. You need to check the details of your existing reservation, for which you remember only the reservation id 'RB9S17'. If and only if further information is required, you can provide additional details. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is mason_johansson_5154. You need to check the details of your existing reservation, for which you remember only the reservation id 'RB9S17'. If and only if further information is required, you can provide additional details. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is juan_li_9671. You need to cancel your current reservation and, instead, book a round-trip flight from Charlotte to Orlando, departing on May 17 and returning on May 19. You prefer economy class and want the flights to be direct. You wish to book for two passengers, Daiki Garcia and Lei Garcia, and add a total of two baggages. You require travel insurance for this trip. You prefer using your certificate for payment, and if necessary, the remaining balance should be charged to your 3086 card. Additionally, you want to update an existing reservation to business class for a trip on May 25, ensuring payment is completed with your 3086 card. You also wish to amend another reservation to include four baggages, out of which two are non-free, using your gift card. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is ivan_garcia_1794. You want to review the details of your existing reservation. You need to update your flights to business class for your reservation, and you want to pay using your credit card ending in 8638. You plan to travel with 2 pieces of checked baggage. Additionally, you are looking to book a new one-way flight from Atlanta to Dallas on May 19 in economy class, with yourself as the passenger. You have one baggage for this trip, and you want insurance. You intend to use your credit card ending in 7155 for this payment. If and only if the previous reservation update cannot be completed, you would cancel another reservation. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is mei_patel_4436. You want to update your reservation for a trip spanning May 19 and 20, ensuring that you fly in economy class. You need to carry 3 bags, all of which are non-free. You prefer to use your credit card to pay for these updates. If and only if your preferred card cannot be used for the flight update, you would like to switch to another card for the baggage charges. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it.",
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
        instruction="Your user id is liam_lee_5870. You are looking to cancel your current reservation for reasons not specified here. You need to find and book a direct flight from Los Angeles to Charlotte on May 17, traveling one way in economy class. You prefer to travel with Raj Hernandez and Sofia Jackson, with their birthdays already in the profile, so you do not prefer to provide them. You have 2 bags, and one of them is a non-free checked bag. You do not want to purchase insurance. You wish to pay using your gift card first, and if it doesn’t cover the full amount, use your 1015550 card for the remaining balance. Additionally, you need to update your existing reservation to include 1 non-free checked bag and change the flights to the ones on May 20. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is daiki_martin_9991. You want to update your flights for the reservation you have, keeping it in economy class, for your trip on May 18. You prefer to use your 7447 card for any additional costs related to this update. You also want to adjust your baggage for this reservation, ensuring you have one checked bag, which you want to pay for using your gift card. Additionally, you need to book a separate one-way flight for Ethan Moore, from Phoenix to Seattle on May 20, in basic economy class. You prefer to use your gift card to cover the cost of this booking. You do not require travel insurance for Ethan's trip. You are reactive to the agent and will not say anything that is not asked. Your birthday is in your user profile so you do not prefer to provide it. If and only if any payment method cannot cover the amount, you are open to suggestions from the agent for alternatives, ensuring the process is smooth and hassle-free.",
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
        instruction="Your user id is ivan_muller_7015. You want to update your reservation for your trip on May 17 to include flights in business class. You are traveling with a total of three bags, one of which incurs an extra fee. You prefer to use your 3563 card to pay for the flight upgrade and your gift card for any baggage fees. Your birthday is in your user profile so you do not prefer to provide it. If and only if your gift card cannot cover the baggage fees, then you would like to use your 3563 card for the remainder. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is mia_lopez_6592. You have a reservation that you want to update to fly from Denver to Atlanta on May 19, in business class. You want to check the reservation details first and then update it if needed. You prefer business class and have a total of 6 baggages, with 1 being non-free. You want to use your gift card for baggage payment. Additionally, you want to book a one-way business class flight for Noah Silva, departing from Denver to Atlanta on May 19. You prefer to use your certificate for part of the payment and pay the remaining balance with your 9314282 card. You also want to add travel insurance for this booking. If and only if the certificate cannot cover the full amount, use your 9314282 card to cover any remaining balance. Your birthday is in your user profile so you do not prefer to provide it. You are reactive to the agent and will not say anything that is not asked.",
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
        instruction="Your user id is ivan_taylor_6615. You want to cancel your current reservation (reservation id not remembered) and book a new one-way flight from Atlanta to Charlotte on May 19. You prefer to fly in economy class and have no baggage for this trip. You want Aarav Kim as the passenger, and since his birthday is in the profile, you do not prefer to provide it. For payment, you wish to use your certificate for the full amount. You also need to update another existing reservation (reservation id not remembered) to include 3 total bags, of which 1 is nonfree, and use your 7447 card for any additional charges. You are reactive to the agent and will not say anything that is not asked. The situation is urgent, so you value efficiency and prompt adjustments.",
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
