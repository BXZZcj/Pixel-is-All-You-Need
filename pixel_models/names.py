"""English names and places for Word Problems and Factual Questions. A fixed split: test names never appear in training."""

import random

PERSONS = """Aaron Abby Adam Adrian Agnes Aiden Alan Albert Alex Alice Alison Amber Amelia Amy Andrew Angela Anna Anne Anthony
April Arthur Ashley Audrey Austin Barbara Ben Benjamin Bernard Beth Betty Bill Blake Bob Bonnie Brad Brandon Brenda Brian
Bruce Bryan Caleb Calvin Cameron Carl Carla Carol Caroline Carter Catherine Cecil Charles Charlotte Chelsea Chloe Chris
Christine Claire Clara Claude Colin Connor Craig Daisy Dale Daniel David Dean Deborah Dennis Diana Diane Dominic Donald
Donna Dorothy Douglas Dylan Eddie Edgar Edith Edward Eileen Elaine Eleanor Elena Eli Elijah Eliza Ella Ellen Emily Emma
Eric Erica Ethan Eugene Eva Evan Evelyn Faith Felix Fiona Frances Frank Fred Gabriel Gail Gary George Gerald Gina Glenn
Gloria Gordon Grace Graham Grant Greg Hannah Harold Harry Heather Helen Henry Holly Howard Hugh Ian Irene Isaac Isabel
Ivan Jack Jacob Jake James Jane Janet Jason Jean Jeff Jenny Jeremy Jerry Jessica Jill Joan Joe Joel John Jonah Jordan
Joseph Joshua Joyce Judith Judy Julia Julian June Justin Karen Kate Keith Kelly Kevin Kim Kyle Larry Laura Lauren Lee
Leo Leon Lewis Liam Lily Linda Lisa Logan Louis Lucas Lucy Luke Lydia Lynn Mabel Marcus Margaret Maria Mark Martha Martin
Mary Mason Matthew Max Megan Melissa Michael Michelle Molly Monica Nancy Naomi Nathan Neil Nicole Nina Noah Nora Norman
Oliver Olivia Oscar Owen Pamela Patrick Paul Paula Peter Philip Phoebe Rachel Ralph Rebecca Rita Robert Robin Roger Rose
Ross Roy Ruby Russell Ruth Ryan Sally Sam Samuel Sandra Sara Sarah Scott Sean Seth Sharon Sheila Simon Sophia Stella
Stephen Steve Susan Sylvia Tara Ted Teresa Thomas Tim Tina Todd Tom Tony Tracy Tyler Ursula Valerie Vera Victor Vincent
Violet Walter Wayne Wendy William Xavier Yvonne Zachary Zoe""".split()

PLACES = [p.replace("_", " ") for p in """Paris Rome London Berlin Madrid Lisbon Dublin Vienna Prague Warsaw Athens Oslo
Stockholm Helsinki Copenhagen Brussels Amsterdam Geneva Zurich Munich Hamburg Cologne Milan Naples Venice Florence Turin
Barcelona Seville Valencia Porto Lyon Marseille Nice Bordeaux Toulouse Glasgow Edinburgh Cardiff Belfast Manchester
Liverpool Leeds Bristol York Oxford Cambridge Budapest Bucharest Sofia Belgrade Zagreb Sarajevo Riga Vilnius Tallinn
Kiev Minsk Moscow Istanbul Ankara Cairo Tunis Algiers Casablanca Lagos Accra Nairobi Kampala Addis_Ababa Dakar
Johannesburg Cape_Town Durban Dubai Doha Riyadh Tehran Baghdad Beirut Amman Jerusalem Karachi Lahore Delhi Mumbai Chennai
Kolkata Bangalore Dhaka Colombo Kathmandu Bangkok Hanoi Manila Jakarta Singapore Seoul Busan Tokyo Osaka Kyoto Beijing
Shanghai Shenzhen Hong_Kong Taipei Sydney Melbourne Brisbane Perth Auckland Wellington Toronto Montreal Vancouver Ottawa
Calgary Boston Chicago Houston Dallas Denver Seattle Portland Phoenix Atlanta Miami New_York Los_Angeles San_Diego
Las_Vegas Detroit Memphis Nashville Orlando Tampa Honolulu Mexico_City Havana Bogota Lima Quito Santiago Caracas
Montevideo Buenos_Aires Sao_Paulo Rio Brasilia La_Paz Panama Kingston Salzburg Innsbruck Graz Basel Bern Lucerne Ghent
Antwerp Rotterdam Utrecht Dresden Leipzig Bonn Stuttgart Nuremberg Bologna Verona Genoa Pisa Palermo Granada Malaga Bilbao
Nantes Strasbourg Lille Reims Aberdeen Dundee Galway Cork Bergen Gothenburg Malmo Tampere Krakow Gdansk Brno Bratislava
Ljubljana Skopje Tirana Nicosia Valletta Reykjavik Luxembourg Monaco Tbilisi Yerevan Baku Almaty Tashkent Ulaanbaatar
Vientiane Yangon Phnom_Penh Kuala_Lumpur Brunei Suva Harare Lusaka Maputo Luanda Kigali""".split()]

JOBS = """teacher doctor nurse farmer baker pilot lawyer writer painter singer actor dancer chef waiter driver sailor
soldier engineer plumber carpenter tailor barber butcher dentist judge banker clerk editor florist librarian""".split()

LAST = """Adams Allen Baker Barnes Bell Bennett Brooks Brown Burns Campbell Carter Clark Collins Cook Cooper Cox Davis Dixon
Edwards Ellis Evans Fisher Foster Fox Gray Green Hall Harris Hayes Hill Howard Hughes Hunt Jackson James Jenkins Johnson
Jones Kelly King Knight Lane Lee Lewis Long Marsh Martin Mason Miller Mills Moore Morgan Morris Murphy Murray Nelson
Owen Palmer Parker Perry Phillips Porter Price Reed Reid Rice Rivers Roberts Rogers Ross Russell Ryan Scott Shaw Simpson
Smith Stone Sullivan Taylor Thomas Turner Walker Ward Watson Webb West White Wilson Wood Wright Young""".split()

COUNTRIES = """France Italy Spain Germany Poland Greece Norway Sweden Egypt Kenya Japan India Brazil Canada Mexico Chile""".split()


def split(names, seed, frac=0.2):
    """-> (train, test): a fixed shuffle; the test names never appear in training"""
    s = sorted(set(names))
    random.Random(seed).shuffle(s)
    k = int(len(s) * frac)
    return s[k:], s[:k]


PERSONS_TRAIN, PERSONS_TEST = split(PERSONS, "persons")
