r"""
tag_taxonomy_seed.py — hand-curated canonical topic taxonomy (tag vocabulary v2).

Read by build_tag_aliases.py, which expands it into tag_taxonomy.csv, tag_aliases.csv and
the legacy-format tag_vocabulary.csv/.parquet. Edit HERE, then rebuild.

Line format:  slug | Display Name | alias; alias; ...
SECTOR_TREE indentation sets the level: 0 = sector_group, 2 = sector, 4 = subsector.
Every slug is unique across ALL families. All 68 legacy sector slugs and 15 legacy
subsector slugs from build_tag_vocab.py (v1) are kept, so old tags stay valid.
Aliases are matched after normalisation (lower-case, & -> and, punctuation -> space,
spelling variants and plurals generated automatically), so list meaningfully different
phrasings only.
"""

SECTOR_TREE = """
grp_financials | Financials | bfsi; financial services; financials; banking financial services and insurance; india financials; finance
  banks_private | Private Banks | private sector banks; private bank; pvt banks; new private banks; bfsi banks private
    small_finance_banks | Small Finance Banks | sfb; sfbs; small finance bank
    payments_banks | Payments Banks | payments bank; payment bank
  banks_public | Public Sector Banks | psu banks; public sector banks; psb; psbs; state owned banks; government banks; nationalised banks
  banks | Banks (General) | banking; banks; banking sector; bank; commercial banks; lenders; bfsi banks; scheduled commercial banks
    cooperative_banks | Cooperative Banks | co-operative banks; urban cooperative banks; ucb
    regional_rural_banks | Regional Rural Banks | rrb; rrbs
  nbfc | NBFCs | nbfc; nbfcs; non banking finance; non banking financial companies; non-banking financial company; non banks; lending; lending and finance; credit services; retail lending
    vehicle_finance | Vehicle Finance | auto finance; vehicle loans; cv finance; tractor finance; two wheeler finance
    gold_loans | Gold Loans | gold loan; gold financiers; gold finance; gold loan nbfc
    msme_lending | MSME Lending | sme lending; msme finance; business loans; msme credit
    consumer_finance | Consumer Finance | consumer lending; consumer lending nbfc; personal loans; consumer loans; credit cards; unsecured lending; consumer durable finance; digital consumer lending
    infra_power_finance | Infrastructure & Power Finance | power finance; infrastructure finance; infra finance; power sector lenders; pfc rec; railway finance
    diversified_nbfc | Diversified NBFCs | diversified financials; diversified financial services; diversified lenders; lending and investments; lending and financial services
    asset_reconstruction | Asset Reconstruction | arc; arcs; asset reconstruction companies; stressed assets; distressed debt
    leasing_finance | Leasing & Rental Finance | leasing; equipment leasing; lease finance; leasing and finance
    loan_against_property | Loan Against Property | lap; secured business loans
  housing_finance | Housing Finance | hfc; hfcs; housing finance companies; home loans; mortgage lenders; mortgages
    affordable_housing_finance | Affordable Housing Finance | affordable hfc; affordable housing finance companies; ahfc
  microfinance | Microfinance | mfi; mfis; microfinance institutions; microlending; jlg lending
  insurance | Insurance | insurers; insurance sector; insurance companies
    life_insurance | Life Insurance | life insurers; life insurance companies; protection insurance; annuities
    general_insurance | General Insurance | non life insurance; general insurers; property and casualty; motor insurance
    health_insurance | Health Insurance | standalone health insurers; sahi; health insurers
    reinsurance | Reinsurance | reinsurers
    insurance_broking | Insurance Broking & Distribution | insurance brokers; insurance aggregators; bancassurance; insurance distribution
  capital_markets | Capital Markets | broking; exchanges; capital market; financial markets; market infrastructure; capital market intermediaries; securities markets
    stock_broking | Stock Broking | brokers; brokerage; discount brokers; retail broking; broking and trading; trading and brokerage
    exchanges_depositories | Exchanges & Depositories | stock exchanges; depositories; clearing corporations; commodity exchanges; exchanges and market infrastructure; mii; market infrastructure institutions
    registrars_rta | Registrars & Transfer Agents | rta; rtas; registrar and transfer agents; kra
    wealth_management | Wealth Management | wealth managers; private wealth; pms; portfolio management services; family offices; private banking
    investment_banking | Investment Banking | merchant banking; ipo advisory; investment banking and advisory; investment banking and brokerage
    credit_rating_agencies | Credit Rating Agencies | rating agencies; credit rating; cras
    alternative_investments | Alternative Investment Funds | aif; aifs; private equity; venture capital; hedge funds; private credit funds
  amc | Asset Management | amc; amcs; asset management companies; mutual fund; mutual funds; fund houses; asset managers; mutual fund industry; investment management
    etfs_funds | ETFs & Index Funds | etf; etfs; exchange traded funds; index funds; index etfs; hybrid etfs; passive funds; gold etf
  fintech | Fintech | financial technology; fin tech; new age financials
    digital_payments | Digital Payments | payments; payment aggregators; payment gateways; upi players; merchant payments; payment services
    digital_lending | Digital Lending | lending platforms; online lending; bnpl; buy now pay later
    insurtech | Insurtech | insurance tech; online insurance platforms
  financial_holding | Financial Holding & Investment Companies | core investment companies; cic; investment holding; investment companies; holding company financials

grp_technology | Information Technology | technology; tech; it; india technology; technology sector
  it_services | IT Services | it; information technology; it services; software services; it and ites; ites; indian it; it outsourcing; software and it services; it consulting; tier 1 it; tier 2 it; midcap it
    erd_services | Engineering R&D Services | er and d; erd; engineering services; product engineering services; digital engineering; engineering research and development
    bpm_bpo | Business Process Management | bpo; bpm; business process outsourcing; kpo; call centres; customer experience management
    cybersecurity_services | Cybersecurity | cyber security; information security; security software; cyber
    cloud_services | Cloud Services | cloud computing; cloud migration services; cloud infrastructure services; managed services
    it_infrastructure_services | IT Infrastructure Services | it infrastructure; system integrators; network integration
    gcc_services | Global Capability Centres | gcc; gccs; global capability centers; captive centres
    geospatial_services | Geospatial & Mapping | geospatial; mapping and navigation; mapping and surveying; geo surveys; gis; geospatial technology; drone survey
  software_products | Software Products | software; saas; software products; enterprise software; software as a service; product companies
    healthcare_it | Healthcare IT | health it; healthtech software; hospital software
    banking_software | Banking & Financial Software | core banking software; fintech software; bfsi software
    gaming_software | Gaming & Interactive Software | game development; gaming studios
    digital_identity_security | Digital Identity & Smart Cards | digital identity; digital signatures; identity solutions; smart cards; e-kyc; authentication
  internet_platforms | Internet & New-Age Platforms | internet; new age; platforms; new age tech; internet companies; consumer internet; online platforms; internet services; digital platforms
    food_delivery | Food Delivery | food tech; online food delivery; foodtech
    online_travel | Online Travel | ota; otas; online travel agencies; travel tech; travel platforms
    online_classifieds | Online Classifieds & Marketplaces | classifieds; online marketplaces; recruitment portals; matrimony portals; real estate portals
    social_media | Social Media & Content Platforms | social networks; content platforms; creator economy
    proptech | Proptech | property tech; real estate tech
  it_hardware | IT Hardware | hardware; computer hardware; technology hardware; servers; pcs; it hardware and equipment
    networking_equipment | Networking Equipment | routers; network switches; network equipment; networking
    office_automation | Office Automation | printers; office equipment; office automation products
    it_distribution | IT Distribution | technology distribution; it products distribution
  semiconductors | Semiconductors | semiconductor; chips; chip; semis; semiconductor industry
    chip_design | Chip Design | fabless; semiconductor design; vlsi design; ic design
    osat_atmp | OSAT / ATMP | osat; atmp; assembly testing marking packaging; chip packaging
    semiconductor_fabs | Semiconductor Fabs | fab; fabs; foundry chips; wafer fabrication
    semiconductor_equipment | Semiconductor Equipment & Materials | wafer equipment; semiconductor materials; semicap
  electronics_manufacturing | Electronics Manufacturing (EMS) | ems; electronics manufacturing; electronic manufacturing services; electronics manufacturing services; contract electronics manufacturing; electronics
    pcb_manufacturing | PCBs | pcb; printed circuit boards; pcba
    electronic_components | Electronic Components | electronic components; passive components; capacitors; connectors; display modules; camera modules
    mobile_manufacturing | Mobile Phone Manufacturing | smartphone manufacturing; handset manufacturing; mobile phones
  data_centers | Data Centres | data centers; data centres; datacenters; data centre; colocation; hyperscale data centres; digital infrastructure; ai data centres
    data_center_reit | Data Centre REITs | data centre reit; data center reit
    ai_infrastructure | AI Infrastructure | ai infra; gpu infrastructure; gpuaas; gpu as a service; ai compute; ai infrastructure providers

grp_healthcare | Healthcare | healthcare; health care; healthcare sector; life sciences
  pharma | Pharmaceuticals | pharma; pharmaceuticals; pharmaceutical; drug makers; indian pharma; pharma companies
    generic_pharma | Generic Pharma | generics; us generics; generic drugs; us generics exporters
    branded_formulations | Branded Formulations | domestic formulations; india branded generics; domestic pharma; ipm; indian pharmaceutical market
    api_bulk_drug | APIs / Bulk Drugs | api; apis; bulk drugs; active pharmaceutical ingredients; kma; pharma intermediates
    specialty_pharma | Specialty Pharma | specialty generics; complex generics; injectables; inhalers; complex injectables
    biosimilars | Biologics & Biosimilars | biologics; biosimilar; biotechnology; biotech; biopharma
    vaccines | Vaccines | vaccine makers; immunisation
    consumer_health | Consumer Health & OTC | otc; over the counter; consumer healthcare; nutraceuticals; wellness products; ayurveda; ayurvedic; herbal products
    animal_health | Animal Health | veterinary; vet pharma; animal healthcare
  cdmo | CDMO / CRDMO | cdmo; crams; contract manufacturing; crdmo; contract development and manufacturing; contract research and manufacturing services
    cro | Contract Research (CRO) | cro; contract research; clinical research; drug discovery services; crdmo research
    peptides_oligos | Peptides & Oligonucleotides | peptides; oligonucleotides; glp-1 cdmo
  hospitals | Hospitals | hospital chains; hospitals and clinics; hospital; multi specialty hospitals; tertiary care
    single_specialty_care | Single-Specialty Care | eye care; ivf; fertility clinics; dental chains; oncology centres; cancer care; maternity hospitals; dialysis
  diagnostics | Diagnostics | pathology labs; diagnostic labs; diagnostic chains; radiology; imaging centres
  medical_devices | Medical Devices | medtech; medical technology; medical equipment; medical devices and supplies; medical consumables; implants; stents; orthopaedic implants
  healthcare_services | Healthcare Services | health services; hospitals and healthcare services; health tech; healthtech; telemedicine
    pharmacy_retail | Pharmacy Retail | pharmacies; pharmacy chains; e-pharmacy; online pharmacy; chemists
  healthcare | Healthcare (General) | healthcare general; medical

grp_consumer_staples | Consumer Staples | consumer staples; staples; consumer non durables; fmcg sector
  fmcg | FMCG | fmcg; fast moving consumer goods; consumer staples; consumer goods; packaged consumer goods; household and personal care; diversified fmcg
    packaged_foods | Packaged Foods | packaged food; food products; biscuits; snacks; confectionery; ready to eat; noodles; bakery
    personal_care | Personal Care | beauty; cosmetics; skin care; hair care; oral care; beauty and personal care; bpc; grooming
    home_care | Home Care | household products; detergents; homecare; home and personal care; cleaning products
    beverages | Non-Alcoholic Beverages | beverages; soft drinks; packaged water; juices; carbonated drinks; bottlers
    alcoholic_beverages | Alcoholic Beverages | alcobev; alcohol; liquor; spirits; beer; imfl; breweries; distilleries; wine
    tobacco | Tobacco & Cigarettes | cigarettes; tobacco products
    dairy | Dairy | dairy products; milk; ice cream; dairy companies
    edible_oils | Edible Oils | edible oil; cooking oil; edible oils and fats; vanaspati
    tea_coffee | Tea & Coffee Brands | tea; coffee; packaged tea
  food_processing | Food Processing | food processing; agro processing; food and agro processing; agri processing; food manufacturing
    rice_grains | Rice, Flour & Grains | basmati rice; rice; flour milling; grains; pulses
    seafood_aquaculture | Seafood & Aquaculture | aquaculture; shrimp; seafood exports; fisheries; marine products; shrimp feed
    poultry_meat | Poultry, Meat & Feed | poultry; animal feed; cattle feed; meat processing; eggs
    specialty_fats | Specialty Fats & Ingredients | specialty fats; food ingredients; starch and derivatives; starch; sweeteners
  sugar | Sugar | sugar mills; sugar and allied products; integrated sugar
    ethanol_distillery | Ethanol & Distilleries | ethanol; ethanol producers; grain ethanol; molasses

grp_consumer_discretionary | Consumer Discretionary | consumer discretionary; discretionary; consumer; consumption; consumer sector; consumer services
  consumer_durables | Consumer Durables | consumer durables; durables; appliances; household durables; white goods sector
    white_goods | White Goods & Appliances | home appliances; household appliances; air conditioners; room acs; refrigerators; washing machines; appliances; kitchen appliances; water purifiers
    consumer_electronics | Consumer Electronics | tv sets; television sets; led tvs; audio; wearables; consumer electronics brands
    consumer_electricals | Consumer Electricals | fans; lighting; switches; electrical appliances; consumer electrical; switchgear consumer; wiring devices
    housewares | Housewares & Kitchenware | kitchenware; cookware; housewares; home products; plasticware
    watches_accessories | Watches & Eyewear | watches; eyewear; accessories; luxury watches
    furniture_home_decor | Furniture & Home Decor | furniture; home furniture; furniture and storage; mattresses and bedding; mattresses; home decor; decor; home goods; foam products
    luggage_travel_goods | Luggage & Travel Goods | luggage; luggage and travel accessories; travel goods; bags
    stationery_writing | Stationery & Writing Instruments | stationery; writing instruments; stationery and art supplies; stationery and gifts; cards and stationery; office supplies; pens
    toys_games | Toys & Games | toys; toys and games; board games
  retail | Retail | retail; retailers; organised retail; retailing; brick and mortar retail
    value_retail | Value Retail | value fashion; discount retail; value retail chains
    grocery_retail | Grocery & Supermarkets | supermarkets; hypermarkets; food and staples retailing; grocery; kirana
    specialty_retail | Specialty Retail | electronics retail; footwear retail; eyewear retail; beauty retail; home decor retail
    apparel_retail | Apparel Retail | fashion retail; clothing retail; apparel retailers
  ecommerce | E-commerce | ecommerce; e-commerce; online retail; internet retail; marketplaces; digital commerce
    quick_commerce | Quick Commerce | q commerce; quick commerce; 10 minute delivery; dark stores
    value_ecommerce | Value E-commerce | value commerce; social commerce; tier 2 ecommerce
    d2c_brands | D2C Brands | d2c; direct to consumer; digital first brands
  textiles | Textiles | textiles; textile; textile manufacturing; textile processing; textiles and apparel; fibers and textiles; fibres and textiles
    spinning_yarn | Spinning & Yarn | spinning; yarn; cotton yarn; spinning mills; cotton spinning; specialty yarns
    weaving_fabrics | Weaving & Fabrics | fabrics; weaving; denim; fabric processing; dyeing
    home_textiles | Home Textiles | towels; bed linen; home furnishings; furnishings; rugs; carpets
    technical_textiles | Technical Textiles | technical textile; geotextiles; industrial textiles
    synthetic_fibres | Synthetic Fibres & Viscose | synthetic fibres; synthetic fibers; polyester; viscose; nylon; man made fibres; mmf; vsf
    garment_exports | Garment Manufacturing & Exports | garments; apparel manufacturing; apparel exporters; garment exporters
    jute | Jute | jute products; jute mills
  apparel_fashion | Apparel & Fashion Brands | apparel; apparel and accessories; apparel and textiles; fashion; branded apparel; clothing; apparel and luxury goods
    innerwear_athleisure | Innerwear & Athleisure | innerwear; athleisure; lingerie; sportswear
    ethnic_wear | Ethnic Wear | ethnicwear; wedding wear
    luxury_fashion | Luxury Goods | luxury; luxury goods; luxury fashion; premium fashion
  footwear | Footwear | shoes; footwear brands; leather; leather products; apparel and footwear
  jewellery | Jewellery | jewellery; jewelry; gems; gems and jewellery; gold jewellery; jewellery and precious metals
    jewellery_retail | Jewellery Retail | jewellery retailers; organised jewellery
    diamonds_gems | Diamonds & Gems | diamonds; cut and polished diamonds; lab grown diamonds; lgd; gemstones
  hotels | Hotels & Hospitality | hotels; hospitality; hotels and hospitality; hotels and resorts; hotels and leisure; lodging; hotels and lodging
  qsr | QSR & Restaurants | quick service restaurants; restaurants; food services; qsrs; casual dining; cafes; hotels restaurants and leisure
  travel_tourism | Travel & Tourism | travel; tourism; travel and tourism; travel and leisure; travel services; leisure travel
  consumer_services | Consumer Services & Leisure | leisure; entertainment and leisure; leisure facilities; amusement parks; theme parks; wellness; fitness; salons
    gaming_betting | Gaming & Betting | gaming; real money gaming; online gaming; casinos; casinos and gaming; betting; esports
    sports_events | Sports & Events | sports; sports management; events; ticketing; live events; sports leagues
    pet_care | Pet Care | pets; pet food
  education | Education | education; edtech; education services; educational services; schools; k12; higher education; test preparation; skilling; coaching
  auto_retail | Automobile Dealerships | auto dealers; dealerships; automobile dealerships; automotive dealership; car dealers

grp_automobiles | Automobiles & Components | automobiles and auto components; auto and auto components; automobile; automobiles; autos; automotive; automobile and auto components
  auto | Automobiles (OEMs) | auto; automobiles; automobile; automotive; autos; auto oems; vehicle makers; oems; automakers
    two_wheelers | Two-Wheelers | two wheelers; 2w; 2ws; motorcycles; scooters; bikes
    four_wheelers | Passenger Vehicles | passenger vehicles; pv; pvs; 4w; cars; suvs; passenger cars
    commercial_vehicles | Commercial Vehicles | cv; cvs; commercial vehicles; trucks; buses; mhcv; lcv
    tractors | Tractors & Farm Equipment | tractors; farm equipment; farm mechanisation; agricultural machinery; harvesters
    three_wheelers | Three-Wheelers | 3w; three wheeler; autorickshaws; e-rickshaws
    bicycles | Bicycles | bicycle; e-bikes; e bicycles
  auto_ancillary | Auto Components | auto ancillary; auto ancillaries; auto components; auto parts; automotive components; automobile components; auto comp; ancillaries; components auto
    tyres | Tyres | tyres; tires; tyre makers; tyre companies
    forgings_castings | Forgings & Castings | forgings; castings; forging; foundry; metal castings; forged components
    powertrain_components | Powertrain & Drivetrain | engine components; transmission; drivetrain; gears; pistons; clutches; brakes; suspension
    auto_electronics | Auto Electricals & Electronics | automotive electronics; auto electricals; wiring harness; sensors; ev electronics; lighting automotive
    auto_interiors | Seating, Interiors & Plastics | seating; interiors; automotive plastics; mirrors; sheet metal
  ev | Electric Vehicles | ev; evs; electric vehicles; e-mobility; electric mobility; ev makers
    ev_charging | EV Charging | charging infrastructure; ev chargers; charging stations
  batteries | Batteries & Energy Storage | batteries; battery; cells; energy storage; battery makers
    lead_acid_batteries | Lead-Acid Batteries | lead acid; automotive batteries; inverter batteries
    lithium_ion_cells | Lithium-ion Cells | li-ion; lithium ion; acc; advanced chemistry cells; cell manufacturing; gigafactory
    bess | Battery Energy Storage (BESS) | bess; battery energy storage system; battery energy storage; grid storage

grp_industrials | Capital Goods & Industrials | industrials; capital goods; industrial; industrial goods; engineering; manufacturing; diversified industrials; india industrials; industrial manufacturing
  capital_goods | Capital Goods | capital goods; capgoods; cap goods; industrial capital goods; heavy engineering
    electrical_equipment | Electrical Equipment | electrical equipment; electricals; heavy electrical equipment; electrical products; switchgear; electrical
    transformers | Transformers | transformer; power transformers; distribution transformers; transformer makers
    wires_cables | Wires & Cables | wires; cables; wires and cables; cable makers; conductors; winding wires; power cables
    power_equipment | Power Equipment | turbines; boilers; gensets; engines and turbines; power plant equipment; btg; power equipment makers
    pumps_valves_compressors | Pumps, Valves & Compressors | pumps; valves; compressors; flow control; pipes and fittings
    bearings | Bearings | bearing; anti friction bearings
    hvac | HVAC & Refrigeration | hvac; commercial refrigeration; air conditioning systems; cooling systems
    industrial_automation | Industrial Automation & Robotics | automation; robotics; industrial robots; automation and control systems; plc; motion control
  industrial_machinery | Industrial Machinery | machinery; industrial machinery; machine tools; industrial equipment; heavy machinery; process equipment; textile machinery; packaging machinery; material handling
    construction_equipment | Construction & Mining Equipment | construction equipment; earthmoving; cranes; mining equipment
  industrial_products | Industrial Products | industrial products; industrial components; metal products; engineering products; industrial supplies
    pipes_tubes | Pipes & Tubes | di pipes; pipes and tubes; ductile iron pipes; line pipes; industrial pipes
    metal_fabrication | Metal Fabrication & Structures | fabrication; steel structures; transmission towers; pre engineered buildings; peb
    fasteners_tools | Fasteners, Tools & Hardware | fasteners; tools; hand tools; tools and hardware; abrasives
    industrial_consumables | Industrial Consumables | welding consumables; graphite electrodes; refractories; carbon products; insulation; welding electrodes; electrodes; welding equipment; sealing and packing; gaskets
    safety_security_equipment | Safety & Security Equipment | safety equipment; personal protective equipment; ppe; protective gear; security equipment; security and surveillance equipment; video surveillance; cctv; fire safety
    precision_components | Precision Engineering & Machining | precision engineering; precision machining; precision manufacturing; precision components; cnc machining
  defence | Defence | defence; defense; defence sector; defence manufacturing; military; defence psus
    defence_electronics | Defence Electronics | defence electronics; defense electronics; radars; avionics; electronic warfare; communication systems defence
    ammunition_explosives | Ammunition & Explosives | ammunition; explosives; commercial explosives; propellants; munitions
    shipbuilding | Shipbuilding | shipyards; naval shipbuilding; ship repair; warships
    drones_uav | Drones & UAVs | drones; uav; uavs; unmanned aerial vehicles; counter drone
    missiles_platforms | Missiles, Aircraft & Platforms | missiles; fighter aircraft; helicopters; armoured vehicles; land systems
  aerospace | Aerospace | aerospace; aerospace and defence; aerospace and defense; aerospace components; aircraft parts
    space_tech | Space Technology | space; space economy; satellites; launch vehicles; space tech; spacetech
    aviation_mro | Aviation MRO | mro; maintenance repair and overhaul; aircraft maintenance
  railways | Railways | railways; rail; indian railways; railway; rail sector; railway infrastructure
    rolling_stock | Rolling Stock & Wagons | wagons; coaches; locomotives; rolling stock; vande bharat; freight wagons
    rail_signalling | Rail Signalling & Safety | signalling; kavach systems; train collision avoidance; rail electronics
    metro_rail | Metro Rail | metro; metro projects; urban rail; rrts

grp_materials | Chemicals & Materials | materials; basic materials; india materials; commodities materials
  chemicals | Chemicals | chemicals; chemical; chemical sector; commodity chemicals; basic chemicals; industrial chemicals; inorganic chemicals; solvents
    petrochemicals | Petrochemicals | petchem; polymers petrochemicals; olefins; aromatics; pta; refining and petrochemicals
    chlor_alkali_soda_ash | Chlor-Alkali & Soda Ash | caustic soda; chlor alkali; soda ash
    industrial_gases | Industrial Gases | industrial gas; oxygen; nitrogen; hydrogen gas suppliers
    carbon_black | Carbon Black | carbon black makers
    dyes_pigments | Dyes & Pigments | dyes; pigments; colourants; dyestuff
  specialty_chemicals | Specialty Chemicals | specialty chemicals; speciality chemicals; specialty chem; spec chem; fine chemicals; performance chemicals
    fluorochemicals | Fluorochemicals | fluorine chemicals; refrigerant gases; hfc gases; fluoropolymers; ptfe
    flavours_fragrances | Flavours & Fragrances | aroma chemicals; fragrances and flavors; flavours and fragrances; f and f
    construction_chemicals | Construction Chemicals | admixtures; waterproofing; tile adhesives
    battery_chemicals | Battery Chemicals & Materials | battery materials; cathode active materials; anode materials; lithium chemicals; electrolyte
    water_treatment_chemicals | Water Treatment Chemicals | water chemicals; ion exchange resins
    adhesives_sealants | Adhesives & Sealants | adhesives; sealants; glues
    surfactants_personal_care_ingredients | Surfactants & Ingredients | surfactants; personal care ingredients; oleochemicals
    agro_pharma_intermediates | Agro & Pharma Intermediates | intermediates; custom synthesis; csm; crams chemicals
    bio_ingredients | Enzymes, Fermentation & Excipients | enzymes; fermentation; fermentation products; gelatin; excipients; microcrystalline cellulose; biochemicals; capsules
  plastics_polymers | Plastics & Polymers | plastics; polymers; plastics and polymers; polymers and plastics; plastic products; rubber and plastics; pvc; resins
    rubber_products | Rubber Products | rubber; rubber goods; belting
  packaging | Packaging | packaging; paper and packaging; printing and packaging; plastics and packaging; packaging materials
    flexible_packaging | Flexible Packaging | flexible packaging; bopp; bopet; packaging films; laminates packaging
    rigid_packaging | Rigid & Plastic Packaging | plastic packaging; rigid packaging; pet bottles; closures; caps and closures; plastic containers
    glass_packaging | Glass Packaging | glass bottles; container glass; pharma glass; glassware
    paper_packaging | Paper Packaging | corrugated boxes; paperboard; mono cartons
  paper | Paper & Forest Products | paper; paper and pulp; pulp; paper products; paper manufacturing; forest products; forest materials; wood products; tissue paper
  cement | Cement | cement; cement makers; cement and building materials; cement sector
  building_materials | Building Materials | building materials; building products; construction materials; home improvement; building material
    tiles_sanitaryware | Tiles, Ceramics & Sanitaryware | tiles; ceramics; sanitaryware; bathware; glass and ceramics; faucets
    wood_panels | Plywood, Laminates & Panels | plywood; laminates; mdf; particle board; wood panels; veneers
    plastic_pipes | Plastic Pipes | pvc pipes; cpvc; plastic pipes and fittings; water tanks
    glass_building | Flat Glass | float glass; architectural glass; flat glass
    fibre_cement | Fibre Cement & Roofing | fibre cement; fibre cement sheets; roofing; roofing materials; asbestos cement; boards
    natural_stone | Granite, Marble & Natural Stone | granite; marble; natural stone; granite and marble; marble and granite; stone products; stone
    concrete_aac_prefab | Concrete, AAC & Prefab | aac blocks; concrete products; ready mix concrete; rmc; formwork solutions; prefabricated buildings; precast
    facades_windows | Facades, Windows & Doors | facade systems; facades; upvc windows; windows; doors; pvc profiles; aluminium windows; curtain walls
  paints | Paints & Coatings | paints; paint; coatings; decorative paints; industrial coatings; paints and coatings

grp_metals_mining | Metals & Mining | metals; metals and mining; mining and metals; minerals and metals; metals_mining; commodities
  metals_steel | Steel | steel; iron and steel; steel makers; steel sector; ferrous; ferrous metals
    integrated_steel | Integrated Steel | integrated steel plants; flat steel; hrc; crc
    secondary_steel | Secondary & Long Steel | long steel; tmt bars; rebars; sponge iron; dri; induction furnace
    stainless_steel | Stainless & Specialty Steel | stainless steel; alloy steel; specialty steel; special steel
    ferro_alloys | Ferro Alloys | ferroalloys; ferro alloys; manganese; silico manganese; ferro chrome
    steel_tubes | Steel Tubes & Pipes | steel pipes; welded pipes; erw pipes; seamless tubes
  metals_nonferrous | Non-Ferrous Metals | non-ferrous; non ferrous; non-ferrous metals; base metals
    aluminium | Aluminium | aluminium; aluminum; alumina; bauxite
    copper | Copper | copper; copper products
    zinc_lead | Zinc & Lead | zinc; lead; silver zinc
    precious_metals | Precious Metals | gold; silver; precious metals; gold refining; bullion
    metal_recycling | Metal Recycling | non-ferrous metals recycling; scrap; secondary aluminium; recycled metals; recycling metals
  mining | Mining | mining; miners; minerals; mineral; mining companies; minerals and mining
    iron_ore | Iron Ore | iron ore mining; pellets
    coal_mining | Coal | coal; coal mining; coal and consumables; thermal coal; coking coal; coal gasification
    gold_mining | Gold Mining | gold mines; gold exploration
    critical_minerals | Critical Minerals & Rare Earths | critical minerals; rare earths; rare earth magnets; lithium mining; graphite; ree
    industrial_minerals | Industrial Minerals | industrial minerals; limestone; silica; barytes; mica
    mining_services | Mining Services | mine developer and operator; mdo; mining contractors

grp_energy | Oil, Gas & Energy | energy; oil and gas; oil gas and consumable fuels; energy sector; hydrocarbons
  oil_gas | Oil & Gas | oil; gas; oil and gas; oil & gas; oilgas; petroleum; o&g; oil gas and consumable fuels; oil and gas services
    upstream_ep | Upstream E&P | e and p; exploration and production; upstream; oil exploration; oil and gas exploration and production
    oilfield_services | Oilfield Services & Equipment | oilfield services; oil and gas equipment; drilling; offshore services; energy equipment and services
    lubricants | Lubricants | lube oil; lubes; lubricant makers
  refining | Refining & Marketing | refining; omc; omcs; oil marketing companies; refiners; refining and marketing; oil refining and marketing; petroleum products; fuel retail
  gas_distribution | Gas Distribution | cgd; city gas; gas distribution; city gas distribution; gas utilities; gas services
    lng_regas | LNG & Regasification | lng; regasification; lng terminals; gas imports
    gas_transmission | Gas Transmission & Trading | gas pipelines; gas transmission; gas trading; oil and gas storage and transportation; pipelines
  new_energy | New Energy (Hydrogen, Biofuels) | new energy; green hydrogen; hydrogen; biofuels; compressed biogas; cbg; saf; sustainable aviation fuel; green ammonia
  energy | Energy (General) | energy general; energy services; energy companies

grp_power_utilities | Power & Utilities | power; utilities; power sector; electricity; power and utilities; industrial utilities
  power_generation | Power Generation | power generation; gencos; generation; ipps; thermal power; power producers; power generation and distribution
    thermal_power | Thermal Power | coal based power; thermal plants; thermal generation
    hydro_power | Hydro & Pumped Storage | hydro; hydropower; hydro power; pumped storage; psp
    nuclear_power | Nuclear Power | nuclear; nuclear energy; small modular reactors; smr
    power_trading | Power Trading & Exchanges | power exchange; electricity trading; iex
  power_transmission | Transmission & Distribution | power transmission; grid; transmission; t and d; t&d; transmission and distribution; power infrastructure; power distribution; discoms
    smart_meters | Smart Meters | smart metering; ami; advanced metering infrastructure
    power_epc | Power T&D EPC | transmission epc; substation epc; t and d epc
  renewables | Renewable Energy | renewables; renewable energy; green energy; clean energy; re; solar; wind; renewable energy equipment; solar energy; solar power; wind energy
    solar_manufacturing | Solar Manufacturing | solar panel; solar panels; solar panel manufacturing; solar pv; solar equipment; solar power equipment; solar pv manufacturing; solar modules; solar cells; solar pv chain; module makers; wafers; polysilicon
    wind_manufacturing | Wind Turbine Manufacturing | wind turbines; wtg; wind equipment
    solar_epc | Solar EPC & Rooftop | solar epc; rooftop solar; solar pumps; c and i solar
    renewable_ipp | Renewable IPPs | renewable developers; solar ipp; wind ipp; hybrid projects; rtc power
  utilities | Utilities (General) | utilities; utility; water utilities; public utilities

grp_infrastructure | Infrastructure & Construction | infrastructure; infra; construction and infrastructure; infrastructure development
  construction | Construction & EPC | construction; epc; engineering and construction; construction and engineering; contractors; epc contractors; engineering procurement construction; construction and development; engineering and project management
    building_construction | Building Construction | buildings epc; civil construction; housing construction
    water_epc | Water EPC | water projects; water supply projects; water infrastructure epc
  infrastructure | Infrastructure | infrastructure; infra developers; infrastructure developers; infra projects; transportation infrastructure
    invits | InvITs | invit; infrastructure investment trusts
    airports | Airports | airport operators; airport infrastructure
    urban_infra | Urban Infrastructure & Smart Cities | smart cities; urban infrastructure; urban development
  roads_highways | Roads & Highways | roads; highways; ham; road construction; national highways; toll roads; expressways; roads and highways; bot projects
  ports | Ports | ports; port operators; ports and terminals; container terminals; shipping and port services
  water_infrastructure | Water & Irrigation | water; water treatment; water management; irrigation; irrigation systems; water supply; desalination; sewage treatment; wastewater

grp_real_estate | Real Estate | real estate; realty; property; real_estate
  realty | Real Estate Developers | real estate; realty; real estate development; property developers; real estate developers; residential; housing; land development; realty development; property development
    residential_realty | Residential Real Estate | residential real estate; housing projects; luxury housing; affordable housing projects
    commercial_realty | Commercial & Office | co working spaces; coworking spaces; managed office spaces; commercial property; commercial real estate; office; offices; grade a offices; commercial office; leasing office
    retail_malls | Retail Malls | malls; shopping centres; retail real estate
    industrial_warehousing | Industrial Parks & Warehousing | industrial real estate; warehousing parks; logistics parks; industrial parks
    reits | REITs | reit; real estate investment trusts; sm reits
    real_estate_services | Real Estate Services | real estate services; property management; brokerage real estate

grp_telecom_media | Telecom & Media | telecom and media; tmt; communication services; communication; telecommunication services
  telecom | Telecom | telecom; telecommunications; telecom services; telecommunication; mobile services; telcos; telecom operators; wireless
    telecom_towers | Telecom Towers | towers; tower companies; telecom infrastructure; fiberisation
    telecom_equipment | Telecom Equipment | telecom gear; telecommunication equipment; optical fibre cable; ofc; 5g equipment
    broadband_fiber | Broadband & Fibre | broadband; fiber to the home; ftth; internet service providers; isp
    satellite_communication | Satellite Communication | satcom; satellite broadband; vsat
  media | Media & Entertainment | media; entertainment; media and entertainment; media entertainment and publication; m and e
    broadcasting | Broadcasting & TV | broadcasting; tv channels; television broadcasters; dth; cable tv
    ott_digital_media | OTT & Digital Media | ott; streaming; digital media; digital content; online video
    films_music | Films & Music | film production; films; movies; music; music labels; multiplexes; cinemas; animation and vfx; visual effects; animation
    print_publishing | Print & Publishing | newspapers; publishing; printing and publishing; books; print media
    advertising_marketing | Advertising & Marketing | advertising; ad agencies; marketing services; outdoor advertising; adtech

grp_services | Services, Logistics & Transport | services; business services; transportation; transportation and logistics; logistics and transportation
  logistics | Logistics | logistics; 3pl; warehousing; supply chain; logistics and warehousing; logistics and supply chain; integrated logistics; freight
    express_courier | Express & Courier | courier; express delivery; parcel; e-commerce logistics
    cold_chain | Cold Chain | cold storage; temperature controlled logistics
    freight_forwarding | Freight Forwarding | freight forwarders; nvocc; customs brokerage
    container_rail | Container Rail & CFS | container freight stations; cfs; icd; rail logistics; ctos; container logistics
    road_transport | Road Transport & Fleet | trucking; roadways; fleet management; bus operators; trucking aggregator; last mile delivery; road freight
  shipping | Shipping | shipping; shipping companies; tankers; dry bulk; container shipping; marine; marine services; maritime
  aviation | Aviation & Airlines | aviation; airlines; air travel; low cost carriers; air cargo
  business_services | Business Services | business services; professional services; consulting; staffing; human resources; hr services; facility management; facilities management; security services; security and investigation services; business support services; outsourcing services
    staffing_hr | Staffing & HR | staffing; temp staffing; recruitment; hr solutions
    facility_management | Facility & Security Services | facility management; integrated facility management; security guards; manned security
    testing_certification | Testing, Inspection & Certification | tic; testing labs; certification
    rental_services | Rental & Leasing Services | rentals; equipment rental; car rental
  environmental_services | Environmental Services & Recycling | waste management; recycling; environmental services; circular economy services; e-waste; plastic recycling; recycled materials
  trading_distribution | Trading & Distribution | trading; distribution; general trading; wholesale trading; import export; trading and distribution; commodity trading; agri trading; exports trading

grp_agriculture | Agriculture & Agri-inputs | agriculture; agri; agribusiness; agri-business; agri business; agri inputs; agri-inputs; farm sector; farming
  agrochemicals | Agrochemicals | agrochemicals; crop protection; pesticides; agricultural chemicals; insecticides; herbicides; fungicides; fertilizers and agrochemicals
  fertilisers | Fertilisers | fertilisers; fertilizers; fertiliser; urea; dap; npk; specialty fertilisers; crop nutrition
  seeds_agri_genetics | Seeds & Agri-Genetics | seeds; hybrid seeds; agri genetics; agri-genetics; seed companies
  agritech | Agritech | agritech; agri tech; farm tech; agri platforms
  plantations | Plantations | plantations; tea plantations; coffee plantations; rubber plantations; tea estates; forestry
  agri_commodities | Agri Commodities & Exports | agri commodities; agri products; agri exports; cotton; spices; agri-commodities; agro commodities

grp_diversified | Diversified & Conglomerates | diversified; conglomerate; conglomerates; multi-sector; multi sector; diversified conglomerates
  conglomerates | Conglomerates | conglomerate; business groups; diversified groups; diversified manufacturing; group companies
  holding_companies | Holding Companies | holding company; holding companies; holdcos; investment holding companies
  other_sector | Other / Unclassified | other; others; unclassified; miscellaneous; misc; na sector

size_segments | Market-Cap Segments (legacy) | market cap segments
  size_bucket | Size Bucket | size
    qib_smallcap | Smallcap | smallcap; small cap; small caps; smallcaps
    midcap | Midcap | mid cap; midcaps; mid caps
    largecap | Largecap | large cap; largecaps; large caps; bluechips
    sme | SME Listed | sme; sme ipo; sme platform; sme stocks; emerge
"""

THEMES = """
capex_cycle | Capex Cycle | capex; capital expenditure cycle; private capex; investment cycle; capex upcycle; capex revival; global capex cycle
govt_capex | Government Capex | public capex; government spending on infrastructure; infra spending; budgetary capex; public investment
capacity_expansion | Capacity Expansion | capacity additions; brownfield expansion; greenfield expansion; new plant; capacity build out
capacity_utilisation | Capacity Utilisation | utilisation; utilization levels; operating rates; plant utilisation
order_book_growth | Order Book Growth | order book; order backlog; order books; order inflow; order intake
order_wins | Order Wins | order win; new orders; contract wins; large deals; deal wins
china_plus_one | China Plus One | china+1; china plus one; c+1; supply chain diversification; supply chain shift from china; friend shoring
pli_scheme | PLI-led Manufacturing | pli; production linked incentive; production-linked incentive; pli beneficiaries
import_substitution | Import Substitution | import substitution; localisation; localization; indigenisation; make in india beneficiaries
export_growth | Export Growth | exports; export opportunity; export momentum; export led growth
premiumisation | Premiumisation | premiumisation; premiumization; premium products; upgrading consumption; trading up
rural_demand | Rural Demand | rural consumption; rural recovery; rural india; rural slowdown
urban_demand | Urban Demand | urban consumption; urban slowdown; metro demand
consumption_recovery | Consumption Recovery | demand recovery; consumption revival; consumption pickup; consumer demand recovery
demand_slowdown | Demand Slowdown | slowdown; weak demand; consumption slowdown; demand weakness; muted demand
margin_expansion | Margin Expansion | operating leverage; margin improvement; ebitda margin expansion
margin_pressure | Margin Pressure | margin contraction; cost pressure; margin compression
pricing_power | Pricing Power | price hikes; pricing actions; price increases
deleveraging | Deleveraging | debt reduction; balance sheet repair; net cash
debt_raise | Fund Raising | debt raise; fund raise; capital raising; qip; rights issue; preferential issue; equity raise; fundraising
working_capital_stress | Working Capital Stress | working capital; receivable days; cash conversion; inventory build up
credit_growth | Credit Growth | loan growth; credit offtake; advances growth; system credit growth
asset_quality | Asset Quality | npa; gnpa; nnpa; slippages; credit costs; bad loans; stressed assets
nim_trend | Net Interest Margin | nim; net interest margin; margins banks; spread compression
deposit_growth | Deposit Growth | deposits; casa; deposit mobilisation; liability franchise; deposit war
unsecured_lending_stress | Unsecured Lending Stress | unsecured stress; personal loan stress; credit card stress; mfi stress; microfinance stress
financialisation_of_savings | Financialisation of Savings | financialization; sip flows; mutual fund flows; retail participation; household savings shift; demat accounts
primary_market_activity | IPO & Primary Market Activity | ipo; ipos; ipo market; ipo pipeline; new listings; primary market; sme ipos
monsoon | Monsoon | monsoon; rainfall; kharif; rabi; el nino; la nina
commodity_inflation | Commodity Inflation | commodity inflation; input cost inflation; raw material inflation; rm inflation; input costs
commodity_deflation | Commodity Deflation | commodity deflation; raw material softening; input cost deflation; lower rm prices
rate_hike | Rate Hikes | rate hike; tightening; hawkish
rate_cut | Rate Cuts | rate cut; easing cycle; dovish; monetary easing
inr_depreciation | Rupee Depreciation | inr depreciation; rupee weakness; weak rupee
gst_change | GST Changes | gst rate cut; gst rationalisation; gst 2.0 impact; gst reform
budget_impact | Union Budget Impact | union budget; budget impact; budget expectations; budget announcements
ev_transition | EV Transition | ev transition; ev adoption; electrification of mobility; ev penetration
renewable_energy | Renewable Energy Build-out | renewable energy; re capacity addition; clean energy transition
energy_transition | Energy Transition | energy transition; decarbonisation; net zero; clean energy; green transition
electrification | Electrification & Grid Capex | electrification; grid capex; transmission capex; power demand growth; t and d capex; power capex
energy_storage | Energy Storage | energy storage; bess; storage tenders; battery storage; pumped storage demand
energy_security | Energy Security | energy security; energy independence; fuel security
data_center_demand | Data Centre Demand | data centre demand; data center capex; hyperscaler capex; dc capacity
ai_adoption | AI Adoption | artificial intelligence; ai; genai; generative ai; agentic ai; ai adoption; llm; machine learning
ai_capex | AI Capex & Infrastructure | ai capex; ai infrastructure; ai infrastructure financing; ai financing; gpu demand; ai compute buildout
genai_it_disruption | GenAI Disruption of IT Services | ai deflation; ai led deflation; ai disruption; ai cannibalisation
digital_transformation | Digital Transformation | digitalisation; digitization; digital adoption; digital india demand
cloud_migration | Cloud Migration | cloud; cloud adoption; cloud spending
cybersecurity_demand | Cybersecurity Demand | cybersecurity; cyber threats; security spending
it_spending_slowdown | IT Spending Slowdown | discretionary tech spend; it demand slowdown; deal ramp downs; bfsi tech spend
semiconductor_localisation | Semiconductor Localisation | semiconductor push; chip manufacturing india; fabs in india; osat india
electronics_localisation | Electronics Localisation | electronics manufacturing push; component localisation; ems growth
defence_indigenisation | Defence Indigenisation | defence indigenisation; atmanirbhar defence; indigenous defence; positive indigenisation list
defence_exports | Defence Exports | defence exports; defense exports
railway_modernisation | Railway Modernisation | railway capex; vande bharat; station redevelopment; kavach rollout; dedicated freight corridors
urban_infra_push | Urban Infra & Metro | metro expansion; urban transport; smart cities push
housing_demand | Housing Demand | housing demand; home sales; residential demand; affordable housing demand
real_estate_upcycle | Real Estate Upcycle | real estate upcycle; property upcycle; housing upcycle; realty upcycle
office_leasing | Office Leasing | office absorption; gcc leasing; flex space; office demand
travel_demand | Travel & Tourism Demand | travel demand; tourism growth; air traffic growth; hotel demand; revpar
hospitality_upcycle | Hospitality Upcycle | hotel upcycle; arr growth; occupancy
quick_commerce_disruption | Quick Commerce Disruption | quick commerce; q-commerce; qcom; dark store expansion; quick commerce disruption
ecommerce_penetration | E-commerce Penetration | online penetration; digital consumption; ecommerce growth
digital_payments_adoption | Digital Payments Adoption | upi adoption; digital payments growth; cashless
financial_inclusion | Financial Inclusion | inclusion; jan dhan; unbanked; financial access
insurance_penetration | Insurance Penetration | insurance penetration; protection gap; under insured
healthcare_access | Healthcare Access | healthcare access; health coverage; medical infrastructure
medical_tourism | Medical Tourism | medical value travel; international patients
us_generics_pricing | US Generics Pricing | price erosion; us price erosion; generic pricing pressure; gx erosion
us_fda_compliance | US FDA Compliance | usfda; fda inspection; warning letter; oai; form 483; import alert; regulatory compliance pharma
patent_cliff | Patent Cliff | patent expiries; loss of exclusivity; loe; blockbuster generics
glp1_obesity | GLP-1 / Obesity Drugs | glp-1; glp1; semaglutide; obesity drugs; weight loss drugs; tirzepatide
cdmo_outsourcing | CDMO Outsourcing | outsourcing to india; biosecure opportunity; cdmo demand
ethanol_blending | Ethanol Blending | ethanol blending; e20; e27; biofuel blending
city_gas_expansion | City Gas Expansion | cgd expansion; png connections; cng stations
nuclear_expansion | Nuclear Power Expansion | nuclear capacity; smr adoption; nuclear opening to private sector
water_infra_push | Water Infrastructure Push | jal jeevan; water supply capex; sewerage projects
road_construction | Road Construction Cycle | highway awarding; road awarding; nhai awarding; road execution
shipbuilding_revival | Shipbuilding Revival | shipbuilding push; maritime capex; ship repair demand
space_economy | Space Economy | space economy; private space; satellite launches; space sector opening
drones_adoption | Drones Adoption | drone adoption; drone demand; counter drone demand
robotics_automation | Robotics & Automation | robotics; automation demand; factory automation
circular_economy | Circular Economy | circular economy; recycling push; epr compliance; waste to value; sustainability recycling
esg_sustainability | ESG & Sustainability | esg; sustainability; green bonds; carbon neutrality; climate disclosure
climate_risk | Climate Risk | climate change; heatwaves; extreme weather; climate risk
tariffs_trade_war | Tariffs & Trade War | tariffs; trade war; reciprocal tariffs; us tariffs; tariff war; trade barriers
fta_trade_deals | Free Trade Agreements | fta; ftas; trade deals; free trade agreement; ceta; trade agreements
geopolitics | Geopolitics | geopolitics; geopolitical; geopolitical risk; war; conflict; middle east; russia ukraine; sanctions
us_policy | US Policy | us policy; trump policies; us administration; us election
china_slowdown | China Slowdown & Dumping | china slowdown; chinese dumping; china overcapacity; china exports dumping
market_share_gain | Market Share Gain | market share gains; share gain; consolidation gains; share from unorganised
new_product_launch | New Product Launches | new launches; product launches; new models; pipeline launches
ma_consolidation | M&A & Consolidation | m&a; m and a; mergers; acquisitions; acquisition; consolidation; buyout; takeover
demerger_restructuring | Demerger & Restructuring | demerger; restructuring; spin off; hive off; merger scheme; corporate restructuring
promoter_pledge | Promoter Pledge | pledge; pledged shares; promoter pledging
promoter_activity | Promoter & Insider Activity | promoter buying; promoter selling; insider trading; insider deals; stake sale; ofs; offer for sale
bulk_block_deals | Bulk & Block Deals | bulk deals; block deals; block; bulk
governance_concern | Governance Concerns | governance concern; red flag; corporate governance; auditor resignation; related party transactions; accounting concerns; governance
earnings_upgrade | Earnings Upgrades | upgrades; eps upgrade; estimate upgrades; beat estimates
earnings_downgrade | Earnings Downgrades | downgrades; eps downgrade; estimate cuts; miss estimates
valuation_rerating | Valuation Re-rating | rerating; re-rating; multiple expansion; valuation upside
valuation_derating | Valuation De-rating | derating; de-rating; multiple compression; valuation concerns; expensive valuations
dividend_buyback | Dividends & Buybacks | dividend; dividends; buyback; buybacks; shareholder returns; payout
turnaround | Turnaround | turnaround story; recovery play; revival
new_age_profitability | New-Age Tech Profitability | path to profitability; ebitda breakeven; cash burn
gold_rally | Gold Price Rally | gold rally; gold prices surge; bullion rally
wedding_season | Wedding & Festive Demand | festive season; festive demand; wedding season; diwali demand
five_g_rollout | 5G Rollout | 5g; 5g rollout; 5g adoption
telecom_tariff_hikes | Telecom Tariff Hikes | tariff hikes; arpu growth; telecom tariffs
labour_codes_impact | Labour Codes Impact | labour codes; wage code; new labour laws
msme_growth | MSME Formalisation | msme; formalisation; formalization; gst formalisation
demographics_theme | Demographics | demographics; demographic dividend; ageing population; young population
aging_population | Aging Population | elderly care; silver economy; senior care
"""

MACRO_FACTORS = """
gdp_growth | GDP Growth | gdp; economic growth; india gdp; gdp growth; growth outlook; gdp methodology
inflation_cpi | CPI Inflation | inflation; cpi; retail inflation; headline inflation; core inflation
wpi_inflation | WPI Inflation | wpi; wholesale inflation
food_inflation | Food Inflation | food prices; vegetable prices; food inflation
monetary_policy | Monetary Policy (RBI) | monetary policy; rbi policy; repo rate; policy rate; mpc; rbi mpc; central banking; rbi
liquidity | Banking Liquidity | system liquidity; liquidity; crr; durable liquidity; omo
bond_yields | Bond Yields | g-sec yields; gsec yields; bond yield; 10 year yield; fixed income; debt markets; rates
fiscal_policy | Fiscal Policy & Deficit | fiscal deficit; fiscal policy; fiscal consolidation; government borrowing; fiscal
tax_collections | Tax Collections | gst collections; tax revenue; direct tax collections; tax buoyancy; taxation
current_account | Current Account & Trade Balance | current account deficit; cad; trade deficit; trade balance; balance of payments; bop; global trade; trade
forex_reserves | Forex Reserves | fx reserves; foreign exchange reserves; forex
currency_inr | Rupee / Currency | inr; rupee; usd inr; currency; currencies; foreign exchange; fx; exchange rate
dollar_index | US Dollar | dxy; dollar index; us dollar strength
crude_oil | Crude Oil | crude; crude oil; oil prices; brent; wti; oil price
natural_gas_price | Natural Gas Prices | gas prices; lng prices; henry hub; natural gas
coal_price | Coal Prices | coal prices; thermal coal prices
metal_prices | Metal Prices | steel prices; aluminium prices; copper prices; lme; base metal prices; metal prices
gold_price | Gold & Silver Prices | gold; gold price; silver price; precious metals prices; bullion
commodity_prices | Commodity Prices | commodities; commodity; commodity prices; soft commodities; agri commodity prices
monsoon_rainfall | Monsoon & Rainfall | monsoon rainfall; imd forecast; reservoir levels; kharif sowing
fii_flows | FII / FPI Flows | fii; fpi; foreign flows; foreign portfolio investors; fii selling; fii buying; capital flows; fund flows
dii_flows | DII & Domestic Flows | dii; domestic flows; mutual fund inflows; sip inflows; domestic institutional investors
fdi_flows | FDI | fdi; foreign direct investment
remittances | Remittances | remittances; nri inflows
us_fed | US Federal Reserve | fed; federal reserve; fomc; us rates; fed rate cut; fed policy
us_economy | US Economy | us economy; us recession; us growth; us gdp; us inflation; us labour market
us_bond_yields | US Bond Yields | us treasury yields; treasuries; ust yields
china_economy | China Economy | china; china economy; chinese economy; china stimulus
europe_economy | Europe Economy | europe; eurozone; ecb policy; uk economy
japan_economy | Japan Economy | japan; boj; yen carry trade
global_growth | Global Growth | global economy; global macro; world economy; global growth outlook; global macroeconomics; global markets macro
global_equity_markets | Global Equity Markets | global markets; global equities; global equity markets; us markets; wall street; emerging markets
geopolitical_risk | Geopolitical Risk | geopolitical tensions; war risk; middle east conflict; russia; iran; israel; strait of hormuz
pmi_manufacturing | Manufacturing PMI | pmi; manufacturing pmi; purchasing managers index
pmi_services | Services PMI | services pmi
industrial_production | Industrial Production (IIP) | iip; industrial output; core sector; core industries
employment | Employment & Jobs | jobs; employment; unemployment; labour market; hiring; payrolls
consumption_indicators | Consumption Indicators | consumption; private consumption; pfce; high frequency indicators; retail sales
rural_economy | Rural Economy | rural wages; farm income; rural economy; msp; mgnrega demand
credit_system | System Credit & Deposits | bank credit; credit deposit ratio; money supply; m3
household_finances | Household Savings & Debt | household savings; household debt; household leverage
electricity_demand | Electricity Demand | power demand; peak demand; electricity consumption
freight_rates | Freight & Shipping Rates | freight rates; container rates; shipping rates; baltic dry index; red sea disruption
auto_volumes | Auto Sales Volumes | auto sales; vehicle registrations; vahan; dispatches
elections_politics | Elections & Politics | elections; politics; state elections; general elections; political risk
corporate_earnings | Corporate Earnings Cycle | earnings growth; corporate earnings; profit cycle; nifty eps; earnings season
market_valuations | Market Valuations | valuations; market valuations; nifty pe; valuation gap; premium to em
market_breadth | Market Breadth & Sentiment | market breadth; sentiment; advance decline; market sentiment; fear and greed
volatility | Volatility | vix; india vix; volatility; market volatility
demographics | Demographics | demographics; population; urbanisation; urbanization
digital_assets | Digital Assets & Stablecoins | crypto; cryptocurrency; bitcoin; stablecoins; digital assets; cbdc; tokenisation
global_liquidity | Global Liquidity | global liquidity; risk on; risk off; carry trade
"""

POLICIES = """
pli_scheme_policy | PLI Scheme (umbrella) | pli; pli scheme; production linked incentive; production linked incentive scheme; pli 2.0; production-linked incentive scheme
pli_electronics | PLI — Mobiles & Electronics | mobile pli; mobile pli 1.0; mobile pli 2.0; pli for mobile phones; pli electronics; large scale electronics manufacturing pli; lsem; pli it hardware
pli_acc_battery | PLI — ACC Batteries | pli acc; pli scheme for acc batteries; acc pli; advanced chemistry cell pli
pli_auto | PLI — Auto & Auto Components | pli auto; auto pli; pli for automobile and auto components
pli_white_goods | PLI — White Goods | pli white goods; pli scheme for white goods; ac and led pli
pli_textiles | PLI — Textiles | pli textiles; pli scheme for textiles; mmf pli
pli_pharma | PLI — Pharma & APIs | pli pharma; bulk drug pli; pli for apis; kms pli
pli_solar | PLI — Solar Modules | pli solar; high efficiency solar pv module pli; pli scheme for solar pv modules; solar pli
pli_food | PLI — Food Processing | pli food; plifpi; pli for food processing
pli_specialty_steel | PLI — Specialty Steel | pli steel; specialty steel pli
pli_drones | PLI — Drones | drone pli; pli drones
make_in_india | Make in India | make in india; make-in-india; make_in_india
atmanirbhar_bharat | Atmanirbhar Bharat | atmanirbhar bharat; aatmanirbhar bharat; atma nirbhar bharat; self reliant india
viksit_bharat | Viksit Bharat 2047 | viksit bharat; viksit bharat 2047; developed india 2047
gst | GST | gst; goods and services tax; gst council
gst_2_0 | GST 2.0 Rationalisation | gst 2.0; gst_2.0; next gen gst; gst rate rationalisation; gst reforms 2025
union_budget | Union Budget | union budget; budget; union budget 2026; union budget 2025; interim budget; finance bill
finance_act | Finance Act / Tax Laws | finance act; finance act 2026; taxation and other laws amendment bill; income tax act 2025; income-tax act; direct tax code; new income tax bill
stt | Securities Transaction Tax | stt; securities transaction tax; capital gains tax; ltcg tax
almm | ALMM (Solar) | almm; approved list of models and manufacturers; almm list ii; almm cells
dcr_solar | Domestic Content Requirement | dcr; domestic content requirement; dcr modules
national_green_hydrogen_mission | National Green Hydrogen Mission | national green hydrogen mission; green hydrogen mission; nghm; sight; sight programme; strategic interventions for green hydrogen transition
pm_kusum | PM-KUSUM | pm-kusum; pm kusum; kusum; pm_kusum; pm kusum scheme; state solar pump schemes; magel tyala saur krushi pump yojana; mukhyamantri saur krushi vahini yojana
pm_surya_ghar | PM Surya Ghar | pm surya ghar; pm surya ghar muft bijli yojana; rooftop solar scheme; pm_surya_ghar; pm surya ghar yojana
national_electricity_plan | National Electricity Plan | national electricity plan; nep 2032; national electricity plan 2032; cea transmission plan
electricity_act | Electricity Act & Rules | electricity act 2003; electricity amendment bill; electricity rules
rdss | Revamped Distribution Sector Scheme | rdss; revamped distribution sector scheme; smart meter scheme
green_energy_corridor | Green Energy Corridor | green energy corridor; gec
energy_storage_obligation | Energy Storage Obligation | energy storage obligation; eso; vgf for bess
viability_gap_funding | Viability Gap Funding | vgf; viability gap funding
national_solar_mission | National Solar Mission | national solar mission; jnnsm; 500 gw target; non fossil target
shanti_act | SHANTI Act (Nuclear) | shanti act; shanti act 2025; nuclear liability amendment; atomic energy amendment
national_coal_gasification_mission | Coal Gasification Mission | national coal gasification mission; coal gasification scheme
carbon_credit_trading | Carbon Credit Trading Scheme | carbon credit trading scheme; ccts; carbon market india
cbam | EU CBAM | cbam; carbon border adjustment mechanism
jal_jeevan_mission | Jal Jeevan Mission | jal jeevan mission; jjm; har ghar jal; jal_jeevan_mission
amrut | AMRUT | amrut; amrut 2.0; atal mission for rejuvenation and urban transformation
namami_gange | Namami Gange | namami gange; clean ganga mission
smart_cities_mission | Smart Cities Mission | smart cities mission; smart city mission
pmay | PM Awas Yojana | pmay; pmay-u; pmay-g; pmay-u 2.0; pradhan mantri awas yojana; pm awas yojana
rera | RERA | rera; real estate regulation act; real estate regulatory authority
pm_gati_shakti | PM Gati Shakti | pm gati shakti; gati shakti; national master plan
national_logistics_policy | National Logistics Policy | national logistics policy; nlp
bharatmala | Bharatmala | bharatmala; bharatmala pariyojana
sagarmala | Sagarmala | sagarmala; sagarmala programme
maritime_vision | Maritime India Vision | maritime india vision 2030; maritime amrit kaal vision 2047; miv 2030
shipbuilding_scheme | Shipbuilding Assistance & Maritime Fund | shipbuilding financial assistance scheme; sbfas; shipbuilding development scheme; maritime development fund; bharat maritime insurance pool; green tug transition programme
national_infrastructure_pipeline | National Infrastructure Pipeline | nip; national infrastructure pipeline; national monetisation pipeline; nmp
udan | UDAN | udan; modified udan; regional connectivity scheme
amrit_bharat_station | Amrit Bharat Station Scheme | amrit bharat station scheme; station redevelopment
kavach | Kavach | kavach; kavach 4.0; automatic train protection
railway_policies | Railway Policies | indian railways freight wagon policy; dedicated freight corridor; dfc; vande bharat programme
india_semiconductor_mission | India Semiconductor Mission | india semiconductor mission; ism; ism 1.0; ism 2.0; semicon india; semicon 2.0; india semiconductor mission 2.0; dli; design linked incentive
ecms | Electronics Component Manufacturing Scheme | ecms; electronics component manufacturing scheme; electronic components manufacturing scheme; specs; spec scheme
indiaai_mission | IndiaAI Mission | indiaai mission; india ai mission; indiaai
digital_india | Digital India | digital india; digital public infrastructure; dpi; india stack
bharatnet | BharatNet | bharatnet; bharatnet phase iii; bharatnet phase-iii
bharat_6g | Bharat 6G | bharat 6g; bharat 6g alliance; 6g vision
dpdp_act | DPDP Act | dpdp act; dpdp act 2023; digital personal data protection act; digital personal data protection act 2023; dpdp rules
ondc | ONDC | ondc; open network for digital commerce
upi | UPI | upi; unified payments interface; upi mdr
unified_lending_interface | Unified Lending Interface | uli; unified lending interface
pm_e_drive | PM E-DRIVE / FAME | pm e-drive; pm e drive; fame; fame ii; fame iii; emps; pm edrive
ev_policy_manufacturing | EV Manufacturing & State EV Policies | spmepci; scheme to promote manufacturing of electric passenger cars; delhi ev policy; state ev policies; delhi ev policy 2.0; delhi ev policy 2026
cafe_norms | CAFE & Emission Norms | cafe iii; cafe norms; corporate average fuel efficiency; bs-vi; bs vi; bs-vii; bs vii; cpcb iv+; cpcb iv; emission norms
remp_scheme | Rare Earth Permanent Magnet Scheme | remp scheme; rare earth magnet scheme; national critical mineral mission; national critical minerals mission; ncmm; critical mineral auctions
ethanol_blending_programme | Ethanol Blending Programme | ebp; ethanol blending programme; e20 mandate; satat; compressed biogas scheme
national_steel_policy | National Steel Policy | national steel policy; steel safeguard; safeguard duty; safeguard_duty; green steel mission
trade_remedies | Anti-dumping & Trade Remedies | anti dumping duty; add; countervailing duty; cvd; bis norms; bis; qco; quality control orders; bis quality control orders
foreign_trade_policy | Foreign Trade Policy & Export Incentives | foreign trade policy; ftp 2023; rodtep; rosctl; export incentives; meis; interest equalisation scheme
pm_mitra | PM MITRA Textile Parks | pm mitra; pm-mitra; pm mitra parks; pm-mitra textile parks; samarth
fta_india_uk | India–UK FTA (CETA) | india-uk fta; uk-india fta; india uk ceta; india-uk ceta; ceta
fta_india_eu | India–EU FTA | india-eu fta; eu-india fta; india eu trade agreement
fta_india_efta | India–EFTA TEPA | tepa; india efta; trade and economic partnership agreement
fta_other | Other Trade Agreements | agoa; usmca; india-us trade deal; bilateral trade agreement; rcep
us_tariff_actions | US Tariff Actions | section 301; section 232; reciprocal tariffs; us tariffs; executive order tariffs; ieepa tariffs
us_legislation | US Legislation | inflation reduction act; ira; one big beautiful bill act; obbba; chips act; chips and science act; genius act; biosecure act; defense production act; dodd-frank act; match act; cybersecurity information sharing act
eu_regulation | EU Regulation | eu ai act; dora; european chips act; eu deforestation regulation; eudr
china_policy | China Industrial Policy | made in china 2025; 15th five-year plan; 14th five-year plan; belt and road initiative; bri; china export controls
epr | Extended Producer Responsibility | epr; extended producer responsibility; battery waste management rules; plastic waste management rules; e-waste rules
montreal_kigali | Montreal Protocol / Kigali Amendment | kigali amendment; montreal protocol; hfc phase down
national_clean_air_programme | National Clean Air Programme | ncap; national clean air programme
bee_norms | BEE Star Rating Norms | bee norms; bee star rating; energy labelling
defence_procurement | Defence Acquisition Procedure | dap 2020; dap 2026; defence acquisition procedure; defence acquisition procedure 2026; dpm 2025; positive indigenisation list; srijan; idex
project_kusha | Project Kusha & Defence Programmes | project kusha; sudarshan chakra; mission sudarshan chakra
ayushman_bharat | Ayushman Bharat / PM-JAY | ayushman bharat; pmjay; pm-jay; ayushman bharat pm-jay; ab-pmjay
abdm | Ayushman Bharat Digital Mission | abdm; ayushman bharat digital mission
cghs | CGHS | cghs; central government health scheme; cghs rates
dpco | Drug Price Control (DPCO/NLEM) | dpco; nlem; drug price control order; nppa
biopharma_shakti | Biopharma SHAKTI | biopharma shakti; shakti biopharma
pharma_quality_rules | Pharma Quality (Revised Schedule M) | schedule m; revised schedule m; gmp norms
nep_2020 | National Education Policy | nep 2020; national education policy 2020; national education policy
pm_kisan | PM-KISAN & Farm Support | pm-kisan; pm kisan; msp; minimum support price; pm-aasha
pmmsy | PM Matsya Sampada Yojana | pradhan mantri matsya sampada yojana; pmmsy; fisheries scheme
pmfme | PM Formalisation of Micro Food Enterprises | pmfme
agri_schemes | Agriculture Schemes | pradhan mantri rashtriya krishi vikas yojana; rkvy; pm dhan dhaanya krishi yojana; mission mausam; digital agriculture mission
mgnrega | MGNREGA / VB-G RAM G | mgnrega; nrega; vb-g ram g; viksit bharat rozgar and ajeevika mission
labour_codes | Labour Codes | labour codes; wage code; wage_code; code on wages; new labour codes
atal_pension_yojana | Social Security Schemes | atal pension yojana; apy; pm jeevan jyoti; pm suraksha bima
eclgs | ECLGS | eclgs; eclgs 5.0; emergency credit line guarantee scheme; ecl scheme
credit_guarantee_schemes | Credit Guarantee Schemes | cgtmse; cgfmu; credit guarantee fund for micro units; credit guarantee scheme
pm_vishwakarma | PM Vishwakarma & MSME Schemes | pm vishwakarma; pm-setu; pm setu; msme schemes; udyam
startup_india | Startup India | startup india; fund of funds for startups; ffs
rbi_regulations | RBI Regulations | rbi circular; rbi guidelines; rbi bank guarantee norms; rbi gold loan harmonization framework; rbi gold loan norms; lcr framework; liquidity coverage ratio; rbi draft directions; rbi master direction
ecl_provisioning | Expected Credit Loss Framework | ecl; expected credit loss; ind as 109 ecl; ecl framework
basel_iii | Basel III | basel iii; basel 3; capital adequacy norms
fcnr_deposits | FCNR(B) & NRI Deposit Schemes | fcnr(b); fcnr (b); fcnr-b; fcnr(b) scheme; fcnr (b) scheme; fcnr(b) deposit scheme; nri deposits
fema | FEMA & Capital Account | fema; liberalised remittance scheme; lrs; fully accessible route; far; ecb; external commercial borrowings; fdi policy; press note 3
ibc | IBC & Insolvency | ibc; insolvency and bankruptcy code; nclt; sarfaesi; sarfaesi act
sebi_icdr | SEBI ICDR (Issue) Regulations | sebi icdr regulations; sebi icdr regulations 2018; sebi issue of capital and disclosure requirements regulations 2018; icdr
sebi_lodr | SEBI LODR (Listing) Regulations | sebi lodr; sebi lodr regulations; sebi lodr regulations 2015; sebi listing regulations; sebi listing obligations and disclosure requirements regulations 2015; sebi (listing obligations and disclosure requirements) regulations, 2015; sebi lodr regulation 30; regulation 30; lodr
sebi_pit | SEBI Insider Trading Regulations | sebi pit; sebi (prohibition of insider trading) regulations, 2015; insider trading regulations
sebi_derivatives | SEBI F&O / Derivatives Framework | sebi equity index derivatives framework; f&o curbs; derivatives framework; weekly expiry rules; closing auction session
sebi_fund_regulations | SEBI Fund Regulations (MF/SIF/REIT/AIF) | sebi reit regulations; sebi invit regulations; sif; sifs; specialised investment funds; sebi mutual fund regulations; sebi aif regulations; sebi fpi consultation paper; sebi fpi rules
sebi_regulations | SEBI Regulations (Other) | sebi; sebi circular; sebi consultation paper; sebi regulations
insurance_reforms | Insurance Reforms | insurance fdi 100; insurance laws amendment; bima sugam; insurance for all by 2047
consumer_protection | Consumer Protection Act | consumer protection act 2019; consumer protection; ccpa
data_centre_policies | Data Centre Policies | karnataka data centre policy; gujarat data centre policy; state data centre policies; data centre policy
state_industrial_policies | State Industrial Policies | state industrial policy; state incentives; state industrial policies
industrial_policy_general | Industrial Policy (Central) | industrial policy; manufacturing mission; national manufacturing mission; industrial accelerator act
"""

SOURCES = """
nuvama | Nuvama | nuvama; nuvama institutional equities; nuvama group; nuvama research; nuvama wealth; nuvama alternative and quantitative research; edelweiss securities
motilal_oswal | Motilal Oswal | motilal oswal; mosl; motilal oswal financial services; mofsl
icici_securities | ICICI Securities | icici securities; i-sec; isec; icici direct; icicidirect; icici_securities
icici_bank_research | ICICI Bank Research | icici bank; icici bank research; icici bank global markets
antique | Antique Stock Broking | antique; antique stock broking
ventura | Ventura Securities | ventura; ventura securities; ventura research
exencial | Exencial Research Partners | exencial; exencial research partners
jefferies | Jefferies | jefferies; jefferies india; jefferies hong kong
sunidhi | Sunidhi Research | sunidhi; sunidhi research; sunidhi securities and finance
axis_capital | Axis Capital | axis capital; axis_capital
axis_securities | Axis Securities | axis securities; axis securities research; axis direct
emkay | Emkay Global | emkay; emkay research; emkay global financial services
arihant | Arihant Capital | arihant; arihant capital; arihant capital markets
prabhudas_lilladher | Prabhudas Lilladher | prabhudas lilladher; pl research; pl capital; phillip lilladher
morgan_stanley | Morgan Stanley | morgan stanley; morgan stanley research; morgan stanley wealth management; morgan stanley investment management
nomura | Nomura | nomura; nomura global markets research
systematix | Systematix | systematix; systematix research; systematix institutional equities; systematix shares and stocks
choice_broking | Choice Broking | choice equity broking; choice broking; choice institutional
clsa | CLSA | clsa
centrum | Centrum | centrum; centrum broking; centrum institutional research; centrum equity research
jm_financial | JM Financial | jm financial; jm financial institutional securities; jmfl
sharekhan | Mirae Asset Sharekhan | sharekhan; mirae asset sharekhan; mirae asset sharekhan research
share_india | Share India Securities | share india; share india securities
equirus | Equirus | equirus; equirus securities; equirus research
three_sixty_one | 360 ONE | 360 one; 360 one capital; 360 one cm research; 360 one capital market; 360 one capital research; iifl wealth
goldman_sachs | Goldman Sachs | goldman sachs; goldman; goldman sachs global institute; goldman sachs asset management; gs research
phillipcapital | PhillipCapital | phillipcapital; phillip capital; phillipcapital india research
yes_securities | YES Securities | yes securities; ysl
anand_rathi | Anand Rathi | anand rathi; anand rathi share and stock brokers; anand rathi research
bofa | BofA Securities | bofa; bofa securities; bofa global research; bank of america; merrill lynch; bofa ml
hdfc_securities | HDFC Securities | hsie; hsie research; hdfc securities; hdfc sec
sbicaps | SBICAPS / SBI Securities | sbicap; sbicap securities; sbicaps; sbi securities; sbi cap
sbi_research | SBI Research | sbi research; sbi ecowrap; state bank of india research
nirmal_bang | Nirmal Bang | nirmal bang; nirmal bang securities; nirmal bang institutional equities
dolat | Dolat Capital | dolat; dolat capital; dolat capital market
incred | InCred Equities | incred; incred equities; incred research; incred capital
hsbc | HSBC | hsbc; hsbc securities; hsbc global research; hsbc private bank; hsbc bank
ambit | Ambit Capital | ambit; ambit capital
front_wave | Front Wave Research | front wave; front wave research
jp_morgan | J.P. Morgan | j.p. morgan; jp morgan; jpmorgan; j p morgan; jpm
bernstein | Bernstein | bernstein; alliancebernstein; sanford bernstein
kantilal_chhaganlal | KC Securities | kantilal chhaganlal; kc securities
iifl | IIFL Capital | iifl; iifl capital; iifl securities; iifl research; iifl capital services
ubs | UBS | ubs; ubs securities
valorem | Valorem Advisors | valorem; valorem advisors
elara | Elara Capital | elara; elara securities; elara capital
kotak | Kotak Institutional Equities | kotak; kotak institutional equities; kotak securities; kie
deven_choksey | DRChoksey | deven choksey; drchoksey; dr choksey
ashika | Ashika | ashika; ashika institutional research; ashika institutional equities; ashika stock services
angel_one | Angel One | angel one; angel broking
deutsche_bank | Deutsche Bank | deutsche bank; deutsche bank research; db research
macquarie | Macquarie | macquarie; macquarie equity research
citi | Citi | citi; citi research; citigroup
dam_capital | DAM Capital | dam capital; dam capital advisors; dam capital research
geojit | Geojit | geojit; geojit investments; geojit financial services
aditya_birla_money | Aditya Birla Capital / Money | aditya birla money; aditya birla capital
smc | SMC Global | smc; smc global securities
avendus | Avendus | avendus; avendus capital; avendus spark; spark pwm
mncl | Monarch Networth Capital | mncl; mncl research; monarch networth capital
bp_equities | BP Equities | bp equities
investec | Investec | investec; investec equities
haitong | Haitong | haitong; haitong international
acmiil | ACMIIL | acmiil; acmiil institutional research
idbi_capital | IDBI Capital | idbi capital; idbi capital research; idbi capital markets and securities
marwadi | Marwadi Financial | marwadi; marwadi shares and finance
wallfort | Wallfort | wallfort; wallfort financial services
definedge | Definedge | definedge; definedge securities
kunvarji | Kunvarji | kunvarji
globe_capital | Globe Capital | globe capital; globe capital market
niveshaay | Niveshaay | niveshaay; niveshaay investment advisors
soic | SOIC | soic; soic research
value_research | Value Research | value research; value research india
et_wealth | Economic Times | the economic times; economic times; et wealth; the economic times wealth; et now
the_ken | The Ken | the ken
the_economist | The Economist | the economist; economist
financial_times | Financial Times | financial times; ft
bloomberg | Bloomberg | bloomberg
cnbc | CNBC | cnbc; cnbc tv18
moneycontrol | Moneycontrol | moneycontrol
business_standard | Business Standard | business standard
swarajya | Swarajya | swarajya
outlook_money | Outlook Money | outlook money
business_india | Business India | business india
mckinsey | McKinsey | mckinsey; mckinsey and company; mckinsey global institute
bcg | BCG | boston consulting group; bcg
bain | Bain & Company | bain; bain and company
kpmg | KPMG | kpmg
pwc | PwC | pwc; pricewaterhousecoopers
ey | EY | ey; ernst and young
deloitte | Deloitte | deloitte
capgemini | Capgemini Research Institute | capgemini; capgemini research institute
blackrock | BlackRock | blackrock; blackrock investment institute
kkr | KKR | kkr
apollo_global | Apollo Global | apollo global management; apollo global
julius_baer | Julius Baer | julius baer; julius baer wealth advisors
bank_of_singapore | Bank of Singapore | bank of singapore
mufg | MUFG | mufg; mufg bank
allianz | Allianz Research | allianz; allianz research
dsp_mf | DSP Mutual Fund | dsp mutual fund; dsp
amc_research_other | Other AMC Research | mutual fund research; amc research
rbi | Reserve Bank of India | reserve bank of india; rbi; rbi bulletin
sebi_source | SEBI | sebi; securities and exchange board of india
ministry_of_finance | Ministry of Finance | ministry of finance; finance ministry; dea
niti_aayog | NITI Aayog | niti aayog; niti aayog and insa
govt_india_other | Government of India (Other) | government of india; ministry; pib; press information bureau
nse | NSE | nse; national stock exchange; national stock exchange of india
bse_source | BSE | bse; bse ltd; bombay stock exchange
imf | IMF | imf; international monetary fund
world_bank | World Bank | world bank; the world bank
adb | Asian Development Bank | asian development bank; adb
bis_source | Bank for International Settlements | bank for international settlements
fsb | Financial Stability Board | financial stability board
wef | World Economic Forum | world economic forum; wef
cfa_institute | CFA Institute | cfa institute
careedge | CareEdge | careedge; careedge ratings; care ratings
crisil | CRISIL | crisil; crisil research; crisil ratings
icra | ICRA | icra; icra research
india_ratings | India Ratings | india ratings; ind-ra; fitch india
transunion_cibil | TransUnion CIBIL | transunion cibil; cibil
dun_bradstreet | Dun & Bradstreet | dun and bradstreet; d and b
company_self | Company (own publication) | company; company presentation; company filing; company website; the company
independent_analyst | Independent Analyst / Blogger | independent analyst; blogger; substack; ca harshad shah; abhishek basumallick; ajay sheth research; koushik pan; the market ear; exponential view; market tide; research ai
source_other | Other / Unknown Source | na; nan; unknown; raw; research_intake; other source
"""

DOC_TYPES = """
single_company_note | Single-Company Note | single company note; company note; analyst note; initiating coverage; company update; result update; stock note; single_company_note
single_company_ar | Annual Report | annual report; single_company_ar; integrated report
single_company_drhp | DRHP / Prospectus | drhp; rhp; prospectus; single_company_drhp; red herring prospectus
single_company_rating | Credit Rating Report | rating rationale; single_company_rating; credit rating report
single_company_policy | Company Policy / Regulatory Filing | single_company_policy; regulatory filing; regulatory_filing; regulatory_filing_other; corporate announcement; corporate_announcement; exchange filing; company policy
multi_company_seminar | Conference / Seminar Notes | multi_company_seminar; seminar; conference notes; investor conference; analyst meet notes; conference takeaways
multi_company_sector | Sector / Thematic Report | multi_company_sector; sector report; sector_report; thematic report; industry report; sector update
govt_policy | Government Policy Document | govt_policy; gov_policy; government_policy; government policy; govt_policy_other; policy_other; government_and_regulatory_other; policy document
macro_report | Macro / Economy Report | macro_report; macro report; macro_other; macro_report_other; economy report; economic update
concall | Concall Transcript | concall; earnings call; conference call transcript
results | Results Filing | results; results_filing; financial results
presentation | Investor Presentation | presentation; investor_presentation; investor presentation; earnings presentation
earnings_review | Earnings Review (multi-company) | earnings review; quarterly results review; results review; q1fy27 results review; earnings season review
earnings_preview | Earnings Preview | earnings preview; results preview; q1fy27 earnings preview; quarterly preview
strategy_report | Equity Strategy / Asset Allocation | india strategy; equity strategy; market strategy; asset allocation; investment strategy; portfolio strategy; top picks; model portfolio; market outlook
technical_report | Technical Analysis Report | technical analysis; technical research; technical calls; charts; derivatives report
daily_market_update | Daily Market Update | daily market update; morning note; market update; daily market; equity market daily update; morning presentation; daily wrap
ipo_note | IPO Note | ipo note; ipo review; ipo analysis; subscribe note
expert_call | Expert Call / Channel Check | expert call; channel check; industry expert; dealer check
meeting_notes | Management Meeting Notes | meeting notes; management meet; management meeting; site visit notes; plant visit
fund_flow_report | Fund Flow / Ownership Report | fund flows report; fii dii flows; mutual fund holdings; ownership changes
commodity_report | Commodity Report | commodity report; commodity outlook; metals outlook
news_article | News / Media Article | news; news article; media article; magazine
book_excerpt | Book / Long-form | book; book excerpt; long form essay
other | Other | other; misc
"""

EVENT_TYPES = """
investor_conference | Investor Conference | investor conference; broker conference; india conference; annual conference; trinity india conference
analyst_meet | Analyst / Investor Day | analyst meet; investor day; capital markets day; analyst day
plant_site_visit | Plant / Site Visit | plant visit; site visit; factory visit
management_meeting | Management Meeting | management meeting; one on one; ceo meeting; management interaction
expert_session | Expert Session | expert call; expert session; industry expert call
earnings_call_event | Earnings Call | earnings call; concall; conference call
agm | AGM / EGM | agm; egm; annual general meeting
roadshow | Roadshow / NDR | roadshow; ndr; non deal roadshow
webinar | Webinar / Panel | webinar; panel discussion; fireside chat
"""

# =========================================================================== round 2
# Added after mining 6,489 real strings (research summaries, company classification,
# policy and source fields). New tags first, then extra aliases for existing tags, then
# ordered regex RULES for long tails (sources, policies, ETFs) that aliases cannot cover.

THEMES += """
regulatory_changes | Regulatory Changes | regulatory; regulatory compliance; regulatory reforms; regulatory reform; regulatory policy; regulations; regulatory updates; compliance; regulatory frameworks; regulatory headwinds; regulatory resets; regulatory transitions; structural reform; reforms; ease of doing business; eodb; public policy; policy; regulatory dispute; regulatory clearance; regulatory advisory; regulatory legal
corporate_actions | Corporate Actions & Disclosures | corporate actions; corporate action; corporate disclosures; corporate announcements; corporate news; corporate updates; corporate update; capital actions; corporate capital actions; disclosures; corporate filings; operational updates; corporate strategy
investment_philosophy | Investing Philosophy & Methods | value investing; growth investing; momentum investing; factor investing; income investing; equity investing; cyclical investing; modern portfolio theory; mpt; mental models; behavioral finance; behavioural finance; investor behaviour; trend following; asset pricing; market efficiency; investment theory; investment methodology; thematic investing; passive investing; portfolio construction; portfolio management; capital allocation; fundamental analysis; free cash flow; pvgo; present value of growth opportunities; equity compounders; value investing philosophy; market history; historical market bubbles; bubble risk; market predictability; rational inattention; decision making under deep uncertainty; dmdu; game theory
derivatives_market | Derivatives & F&O Activity | derivatives; equity derivatives; derivatives analysis; rollover; market rollover analysis; expiry rollover analysis; rollover report; f and o; futures and options; derivatives market analysis; systematic flows; cta; options activity
management_change | Management Change | management change; leadership transition; ceo change; kmp resignation; management exit; succession
exchange_surveillance | Exchange Surveillance | surveillance; asm; gsm; esm; t2t; trade to trade; additional surveillance measure
"""

MACRO_FACTORS += """
macro_general | Macroeconomy (General) | macroeconomics; macro; macro economics; macro economy; macro outlook; economy; macroeconomic; india economy; indian economy; macroeconomic trends; macroeconomic policy; macro trends; macroeconomic indicators; macro economic indicators; macro update; macro updates; macroeconomic outlook; macro economic outlook; economic policy; economics; macro analysis; macro roundup; india macro; indian macro; economic studies; macro policy; macroeconomic reforms; macroeconomic review; macroeconomic overview; macroeconomic update; macroeconomic analysis; macroeconomic strategy; macro economic; macro tidings; india economic monitor; macro economic trends; economic methodology; macro political environment; geoeconomics
indian_equity_market | Indian Equity Market | equity markets; indian equities; indian equity market; indian equity markets; india equities; equities; equity; markets; market; equity market trends; indian markets; market commentary; market analysis; market insights; indian equities market overview; equities india; market repricing; market resilience; smid cap equities; smids; small; micro cap; market capitalisation segment wise analysis; corporate india; indian corporate
credit_markets | Credit Markets | credit markets; credit; corporate bonds; debt trends; borrowing; debt refinancing; debt; credit risk; credit underwriting risk; nbfi risk; credit access
"""

DOC_TYPES += """
corporate_filings_digest | Corporate Filings Digest | daily corporate filings; corporate filings digest; disclosures digest; india corporate disclosures; multi sector corporate disclosures; daily transaction digest; exchange filings digest
"""

POLICIES += """
coal_policy | Coal Policy (linkage, auctions, gasification) | shakti; shakti policy; shakti linkage; coal block auctions; 1 bt coal production push; scheme for promotion of surface coal lignite gasification projects; cmpfo
upstream_oil_gas_policy | Upstream Oil & Gas Policy | oalp; open acreage licensing policy; help; hydrocarbon exploration and licensing policy; dsf bid round; oilfields regulation and development amendment act; national deep water exploration mission; samudra manthan; samudra manthan mission; samudra mantan; windfall tax
gas_policy | Gas Policy & Grid | one nation one gas grid; north east gas grid; natural gas supply regulation order; png drive; pngrb; pngrb smart gas meter mandate; city gas distribution bidding; gas pricing reform; kkmbpl pipeline
telecom_policy | Telecom Policy | telecommunications act 2023; telecommunications act; telecom reforms package; agr relief scheme; national broadband mission; national broadband mission 2.0; trai regulations; spectrum auction
it_cyber_law | IT & Cyber Law | it act 2000; information technology act 2000; it rules 2021; information technology rules 2021; cert in directions; cyber security regulations; bharatiya nyaya sanhita; bharatiya nagarik suraksha sanhita; bharatiya sakshya adhiniyam
pay_commission_pension | Pay Commission & Pensions | 8th pay commission; 8th_pay_commission; pay commission award; national pension system; nps; old pension scheme; unified pension scheme; epfo; pradhan mantri shram yogi maandhan
health_policy | Health Policy | national health policy; national health policy 2017; ayush; jan aushadhi; pradhan mantri bhartiya janaushadhi pariyojana; national medical device policy; drugs and cosmetics act; national pharmacy commission bill; national policy for rare diseases; mental healthcare act; national immunization programmes; heal in india; nps swasthya; surrogacy act; art regulation act
company_law | Company Law & Accounting | companies act 2013; companies act; ind as; jan vishwas act; jan vishwas amendment of provisions act; csr rules
real_estate_regulation | Real Estate Regulation & Urban Planning | model tenancy act; dcpr 2034; udcpr; maharashtra housing policy; delhi master plan 2047; mpd 2047; urban land ceiling and regulation act; mofa 1963; rera act 2016; swamih; swamih 2.0; dharavi redevelopment project; registration bill
state_excise_policy | State Excise & Liquor Policy | excise policy; up excise policy; karnataka excise reform; tasmac policy; karnataka aib taxation; liquor policy
online_gaming_regulation | Online Gaming Regulation | online gaming bill 2025; online gaming act; gaming regulation
tourism_policy | Tourism Policy | swadesh darshan scheme; farm stay policy; cruise bharat mission; tourism policy; coastal malnad karnataka tourism policy
food_commodity_controls | Food & Commodity Controls | open market sales scheme; omss; price stabilisation fund; essential commodities act; national food security act; pmgkay; pradhan mantri garib kalyan anna yojana; sugar control order; sugar stockholding norms; stock limits
water_resources_policy | Water Resources Policy | national water policy; river linking; brahmaputra basin master plan; jal marg vikas project; water act; flood protection act
fertiliser_subsidy_policy | Fertiliser Subsidy & Urea Policy | nutrient based subsidy; nbs; nbs subsidy; nutrient based subsidy scheme; new investment policy for urea; national investment policy for urea; nipu; fertilizer subsidy scheme; urea policy; pm pranam scheme
rural_infra_schemes | Rural Infrastructure & Welfare Schemes | pmgsy; pradhan mantri gram sadak yojana; pmgsy iv; ddugjy; saubhagya; rggvy; rajiv gandhi grameen vidyutikaran yojana; vibrant villages programme; pm svanidhi; pmjdy; one district one product; odop
payments_policy | Payments Regulation | bbps; bharat connect; payments vision 2028; payment infrastructure development fund; pidf; payment and settlement systems act; upi monetisation policy; aeps; ocen; national payments vision
ai_policy_global | AI Policy (Global) | ai act; america's ai action plan; ai opportunities action plan; national ai agenda; national policy framework for ai; ai basic plan; genesis mission; rbi draft guidance on ai; free ai; guidelines on ai risk management; sound practices for responsible adoption of ai
monetary_policy_framework | Monetary Policy Framework | inflation targeting; flexible inflation targeting; cash reserve ratio; crr; market stabilisation scheme; operation twist; qe; quantitative easing; rbi monetary policy; mpc monetary policy; fed monetary policy; statement on longer run goals and monetary policy strategy; standing deposit facility
geopolitical_treaties | Treaties & Geopolitical Agreements | aukus; new start; chemical weapons convention; biological and toxin weapons convention; arms trade treaty; unclos; taiwan relations act; monroe doctrine; shanghai cooperation organisation; brics; brics 2026; quad; us iran agreement; paris agreement; paris climate accord; kyoto protocol
foreign_policy_other | Foreign Government Programmes | saudi vision 2030; vision 2030; vision 2050; france 2030; deutschlandfonds; digital morocco 2030; skillsfuture; corporate value up program; iktva; in kingdom total value add program; national water strategy 2030; rearm europe; repowereu; european green deal; horizon europe
textile_schemes | Textile Schemes | national technical textiles mission; nttm; technical textile mission; national fibre scheme; textile expansion and employment scheme; weavers comprehensive welfare scheme; pli for technical textiles
defence_programmes | Defence Programmes | amca; advanced medium combat aircraft; akashteer; agnipath; make ii; make 2; make two; greenfield helicopter project
space_programmes | Space Programmes | gaganyaan; artemis accords; nasa artemis; national security space launch; landsat; copernicus
"""

SOURCES += """
dalal_broacha | Dalal & Broacha | dalal and broacha; dalal and broacha stock broking
swastika | Swastika Investmart | swastika; swastika investmart
way2wealth | Way2Wealth | way2wealth; way2wealth brokers
bajaj_broking | Bajaj Financial Securities | bajaj financial securities; bajaj broking
skp_securities | SKP Securities | skp securities; skp research
keynote_capitals | Keynote Capitals | keynote capitals
hem_securities | Hem Securities | hem securities; hem securities research
lkp | LKP Securities | lkp; lkp research; lkp securities
gepl | GEPL Capital | gepl; gepl capital
smifs | SMIFS | smifs; smifs limited; smifs techno funda
b_and_k | B&K Securities | b and k securities; batlivala and karani
kamayakya | Kamayakya Wealth Management | kamayakya; kamayakya wealth management
sushil_finance | Sushil Financial Services | sushil financial services; sushil finance
prudent_corporate | Prudent Corporate Advisory | prudent corporate advisory; prudent corporate advisory services
indus_equity | Indus Equity Advisors | indus equity advisors
barclays | Barclays | barclays; barclays capital; barclays private bank
standard_chartered | Standard Chartered | standard chartered; standard chartered bank
credit_suisse | Credit Suisse | credit suisse
mizuho | Mizuho | mizuho; mizuho securities
bnp_paribas | BNP Paribas | bnp paribas; bnp paribas wealth management
dbs | DBS | dbs; dbs chief investment office
uob | UOB | uob; uob global economics and markets research
franklin_templeton | Franklin Templeton | franklin templeton; franklin templeton mutual fund; franklin templeton asset management
icici_pru_mf | ICICI Prudential MF | icici prudential mutual fund; icici prudential amc
sbi_mf | SBI Funds Management | sbi funds management; sbifm research; sbi mutual fund
axis_mf | Axis Mutual Fund | axis mutual fund
redseer | Redseer | redseer; redseer strategy consultants
property_consultants | Property Consultants | jll; jll research; knight frank; colliers; savills; savills india research; anarock; cbre; proptiger
sp_global | S&P Global | s and p global; s and p
moodys | Moody's | moody's; moodys; moody's ratings
msci | MSCI | msci; msci inc
screener_tools | Data Platforms (Screener, Trendlyne ...) | screener in; screener.in; trendlyne; tijori; tijori stack; stockscans; pms bazaar; pmsbazaar; fundsindia
reuters | Reuters | reuters
wsj | Wall Street Journal | wall street journal; wsj
barrons | Barron's | barron's; barrons
forbes | Forbes | forbes; forbes india
mint | Mint | mint; livemint; livemint com
hindu_businessline | Hindu BusinessLine | the hindu business line; businessline; businessline portfolio; hindu business line
business_today | Business Today | business today
capital_market_pub | Capital Market (publication) | capital market publishers; capital market publishers india
media_other | Other Media & Newsletters | the information; wired; tom's hardware; visual capitalist; inc42; moneyweek; moneylife; pcquest; nhk news; the epoch times; the wrap; thewrap
think_tank_academic | Think Tanks & Academia | brookings; hudson institute; rand corporation; csis; center for strategic and international studies; sipri; kiel institute; ncaer; centre for social and economic progress; csep; hinrich foundation; rhodium group; springer nature; palgrave macmillan; edward elgar publishing; harvard business review; stripe press; congressional research service
foreign_govt_regulator | Foreign Governments, Regulators & Central Banks | european central bank; federal reserve bank of new york; deutsche bundesbank; banque de france; de nederlandsche bank; esma; european securities and markets authority; financial conduct authority; iosco; us securities and exchange commission; u s energy information administration; u s geological survey; white house; tokyo stock exchange; cboe; european space agency; mission of china
govt_india_bodies | Indian Government Bodies | parliament of india; rajya sabha; economic advisory council to the pm; eac pm; central board of direct taxes; cbdt; department of atomic energy; pib delhi; government of gujarat; gujarat maritime board; maharashtra government gazette; nclt; securities appellate tribunal; national institute of securities markets; nsdl; cdsl
industry_association | Industry Associations & Conferences | fada; federation of automobile dealers associations; nasscom; dsci; data security council of india; assocham; ficci; cii; world gold council; coffee board; bharath digital infrastructure association; investor connect summit
consulting_firm | Consulting Firms | arthur d little; roland berger; accenture; ibm institute for business value; counterpoint; jmk research; 1lattice; eminence strategy consulting; strategic foresight guild
pms_aif_house | PMS / AIF / Fund Houses | abakkus; valuequest; sahasrar capital; trinetra asset managers; aequitas; aurum capital; ikigai asset management; sageone; ionic wealth; toro wealth managers; 3p investment managers; first principles asset services; sanctum wealth; invexa capital; monarch aif; ankur capital; whiteoak capital amc; edelweiss mutual fund; nippon india alternative investments; dsp asset managers; amova asset management; amundi; amundi investment institute; state street investment management; gqg partners; oaktree capital management; alta fox capital; palliser capital; sycamore tree capital; k2 investment management; man ahl; fidelity international; horizon management; novus capital advisors
foreign_broker_other | Other Global Brokers & Banks | panmure liberum; william blair; bca research; ts lombard; greed and fear; houlihan lokey; natixis; handelsbanken; mufg securities; mirae asset securities; china construction bank; industrial and commercial bank of china; bofa global quantitative strategy; ubs gwm; ubs global wealth management; clsa limited
indian_bank_research | Indian Bank Research | bank of baroda research; axis bank research; axis bank; union bank of india; state bank of india
broker_other | Other Indian Brokers & Advisors | stock broking; share broking; investmart; finserve; research analyst
"""


# Extra aliases for EXISTING tags (slug | alias; alias ...), found in the mined strings.
EXTRA_ALIASES = """
pharma | api and formulations; api and formulations manufacturers; api formulation manufacturers; apis and formulations; formulations; biopharmaceutical companies; anti microbial resistance; amr; cns drug development; human health
api_bulk_drug | api manufacturers; steroid api manufacturers; fermentation based apis; bempedoic acid value chain; biopharmaceutical api
biosimilars | biopharmaceuticals; biopharmaceutical; biologicals pharma
consumer_health | otc drugs; otc products; dietary supplements; nutritional supplements; nutritional products; nutritional supplement manufacturers; health foods; herbal supplements; pain management; pain management products; contraceptives; indian nutrition
specialty_pharma | oncology; oncology drugs; ophthalmics
cro | drug discovery; r and d drug discovery; cro testing; contract development
medical_devices | dental products; surgical equipment; surgical instruments; surgical supplies; surgical consumables; surgical suture manufacturers; sutures; labware; labware manufacturers; ophthalmic lenses; disposable medical products
diagnostics | pathology laboratories; pathology lab
healthcare_services | digital health
single_specialty_care | dental care
packaged_foods | namkeen; peanut products; frozen foods; frozen food processors; processed foods; processed food companies; processed food manufacturers; fmcg food; ready to cook; salt
food_processing | fruit processors; fruits and vegetables; agro food processors; agro products and oils
rice_grains | flour mills; rice processors
poultry_meat | broilers; egg products; fresh protein
edible_oils | soybean oil; soybean oil processors; specialty oils; oil extraction; oil extractors
specialty_fats | cocoa butter equivalents
alcoholic_beverages | beer manufacturers; imfl manufacturers; wineries; alcoholic
tobacco | bidis
home_care | hygiene products; hygiene and health products; sanitation products; agarbatti; agarbatti manufacturers; agarbatti and aroma products; disposable products
personal_care | baby care; baby products; sanitary napkins and hygiene; feminine care; soaps
agri_commodities | farm products; agro products; floriculture; castor oil
seafood_aquaculture | seafood processors
plantations | tea production
white_goods | air coolers; air cooler manufacturers; sewing machines
consumer_electricals | lamps; wiring; wiring manufacturers
housewares | tableware
apparel_fashion | womenswear; knitwear; knitwear manufacturers; embroidery
synthetic_fibres | acrylic fibre; acrylic fibre manufacturers; acrylics; filament yarn; synthetic yarn; synthetic yarns; synthetic yarn manufacturers; silk and synthetics; wool and synthetics
spinning_yarn | yarn manufacturers; silk
footwear | tanning
retail | department stores
qsr | qsr chains; catering
online_classifieds | online matrimony
hotels | resorts; clubs and resorts; adventure resorts; vacation ownership
travel_tourism | tour operators; tour; ropeways; ropeway operators
consumer_services | photography; photographic products; art and collectibles; collectibles
education | educational institutions; management institutes; it training; it training providers; vocational training; it and vocational training; online learning; edutech; upskilling; corporate upskilling
wealth_management | investment advisors; investment advisory; wealth advisory; portfolio managers; global wealth; global wealth trends
investment_banking | merchant banks; investment banks; corporate advisory
financial_holding | investment; investments; credit and investments; credit and investments nbfc; nbfc investment; investment trusts; investment and management
banks | regional banks; indian banks; south based banks
vehicle_finance | vehicle financing
leasing_finance | asset financing; alternative financing
registrars_rta | registry and transfer agency
amc | listed amcs; equity funds; hybrid funds; equity savings funds; active long short funds; hybrid long short funds; arbitrage funds; fund management; asset mgrs
alternative_investments | alternative assets; private credit
credit_rating_agencies | credit bureau
digital_payments | cash management; loyalty programs; digital vouchers; treds; trade receivables discounting system
capital_markets | securities; forex; slb; securities lending
it_services | ai and analytics; ai and machine learning; ai solutions; analytics; data analytics; data analytics and research; data management; application development; mobile applications; e governance; e governance solutions; govtech; iot; erp services
software_products | erp; enterprise resource planning; regtech; it saas
bpm_bpo | bpo and analytics
advertising_marketing | digital marketing; public relations; signage; signage manufacturers
films_music | content creation; content production; film and studio production; movie production; tv production; vfx studios; vfx
broadcasting | radio
print_publishing | printing; book printing; commercial printing; gravure printing; printing inks
business_services | consultancy; market research
staffing_hr | executive search; future of work; workplace
facility_management | building maintenance
environmental_services | green tech solutions; carbon credits; rpet; recycled pet; clean tech
new_energy | bioenergy; bioenergy producers; biomass; bioeconomy
electronic_components | magnets; ferrites; microelectronics; display equipment; opto mechatronics
electronics_manufacturing | ems providers; esdm
mobile_manufacturing | mobile devices; mobile handset manufacturers; smartphones
osat_atmp | assembly and testing
semiconductors | memory; dram memory; ai memory; processors; co packaged optics; cpo; memory chips
semiconductor_equipment | lithography equipment; cleanrooms
chip_design | gpu compilers
telecom | mobile network operators
telecom_equipment | fibre optics; optical networking; optical fibre; optical fiber
networking_equipment | network solutions; private networking
electrical_equipment | insulators; rectifiers; converters; laminations; electric motors; control systems; grid equipment; electrochemical equipment
power_equipment | engines; engine manufacturers; combustion equipment; combustion equipment manufacturers; heat exchangers; heat exchanger manufacturers
pumps_valves_compressors | pumps manufacturers; fluid control; fluid control equipment; hydraulic equipment; pneumatic equipment; hydraulic fittings; hydraulics; pneumatics
hvac | air filtration; heating and cooling; heating equipment; refrigeration equipment; thermal management
industrial_automation | instrumentation; instruments; scientific instruments; gauging equipment; metering; 3d printing; industry 4 0
industrial_machinery | cnc machines; specialty equipment; separation equipment; cryogenic equipment; glass lined equipment; glass lined equipments; conveyor systems; lifting equipment; elevators and escalators
fasteners_tools | cutting tools; tool manufacturers; tools manufacturers; chains and sprockets
powertrain_components | auto gears; gears and gearboxes; crankshafts; leaf springs; leaf spring manufacturers; auto axle manufacturers; wheels; wheels and rims; emission control systems
auto_electronics | autonomous driving; ev motor control units
tyres | tire manufacturers; tyre manufacturers; tyre retreading
ev | ev ecosystem; ev systems; ev mobility
internet_platforms | shared mobility; commercial mobility; mobility solutions; ride hailing
rubber_products | conveyor belts; latex products; synthetic latex; synthetic latex manufacturers
plastics_polymers | masterbatches; plasticizers; plasticizer manufacturers; pvc compounds; thermoplastics; thermoplastics processors; polypropylene; elastomers; composites; biocomposites
specialty_chemicals | catalysts; activated carbon; activated carbon manufacturers; additives
flavours_fragrances | flavours; fragrances; flavors and fragrances
chemicals | amines; amines manufacturers; caprolactam; caprolactam producers; peroxides; phosphates; hypochlorite; borax and derivatives; vinyl acetate monomer; propylene oxide derivatives; cellulose
surfactants_personal_care_ingredients | castor derivatives; castor oil derivatives; fatty acids and derivatives; esters and derivatives
dyes_pigments | dye manufacturers; dyes manufacturers; pigments
flexible_packaging | bopp films; specialty films
packaging | labels; fibc bags; drums and barrels; strapping solutions; straps
wood_panels | panels; surfaces; wood composites; vinyl flooring; flooring; flooring manufacturers; vinyl products
tiles_sanitaryware | vitrified tiles; bath fittings; plumbing fixtures
glass_building | glass; glass products; glass manufacturers; toughened glass; crt glass
industrial_products | alloy products; gas cylinders; iron products
secondary_steel | pig iron; pig iron manufacturers; wire rods; wire ropes; wire rope manufacturers; wire products; iron
stainless_steel | alloys; bimetal strips
wires_cables | wire manufacturers; wire industry
critical_minerals | chrome ore; ilmenite and rutile; chromium and vanadium derivatives; uranium
industrial_minerals | magnesite
mining_services | mine planning; mine planning and design
metal_recycling | ship breaking
realty | land assets; integrated township development
commercial_realty | commercial developers; commercial development; flexible workspace
retail_malls | shopping malls; shopping mall operators
reits | singapore reits
roads_highways | roads and bridges
ports | port development; port operations; terminals; cargo handling; dredging
shipping | offshore support vessels
express_courier | express cargo
construction | mep solutions; turnkey contracts
power_transmission | hvdc
missiles_platforms | 155mm artillery; armaments; global arms transfers
space_tech | earth observation
agrochemicals | biologicals; agri biologicals
agritech | agtech
seeds_agri_genetics | seed manufacturers; seed producers
tractors | agricultural equipment; agricultural implements
conglomerates | kirloskar group
upstream_ep | offshore oil; exploration
refining | o2c strength; integrated oil; refineries
gas_distribution | gas transition; gas strategy; gas support
fertilisers | micronutrients
etfs_funds | index; broad market; broad market etf; broad market etfs; sectoral; sectoral etf; sectoral etfs; sector etfs; thematic etfs
robotics_automation | humanoid robots; humanoids; humanoid; physical ai; embodied ai; autonomous systems; world models; spatial intelligence
ai_adoption | ai theme; ai transformation; ai productivity; ai innovation; ai ecosystem; ai enablers; ai tailwinds; ai decision intelligence; ai in investment research; superintelligence; ai risk; ai ethics; ai risk management; frontier tech; tech acceleration; innovation; india ai complex
ai_capex | ai buildout; ai supercycle; neocloud; ai scarcity; ai memory bottleneck; chipflation; ai training
digital_transformation | digital economy; digital; digital sovereignty
cybersecurity_demand | cybertech; digital forensics; forensic science
governance_concern | fraud; forensic accounting; forensic analysis; accounting
promoter_activity | insider
debt_raise | capital raise; share issuance; preferential share issue
ma_consolidation | deal activity; joint venture; voluntary delisting
esg_sustainability | environment; environmental policy; emission
climate_risk | climate policy; weather
aging_population | longevity; longevity economy
primary_market_activity | spacex ipo
glp1_obesity | obesity
corporate_earnings | earnings update; earnings outlook; us earnings; corporate profit to gdp ratio
gold_price | gold wealth effect; global gold market; gold reserves; central bank reserves
bond_yields | interest rates; interest rate; macro interest rates; sovereign bonds
fiscal_policy | sovereign debt; sovereign debt dynamics; tax policy
currency_inr | fx markets; fx impulse
monetary_policy | monetary; monetary systems
global_growth | recession risks; global imbalances; global cost of living; global economic forecasts; global scenarios; global competitiveness; global competition; global
europe_economy | germany; european equities
china_economy | rmb internationalization; rmb internationalisation
japan_economy | yen carry
global_equity_markets | asia pacific markets
us_economy | us economic outlook
strategy_report | strategy; macro strategy; strategy update; multi asset strategy; multi asset investment outlook; equity outlook; india investment thesis; high conviction picks; top investment picks; global investment; sector rotation; market rotation; defensives; equity research; india equity research; outlook; fy27 outlook; equity market outlook; global macro strategy
daily_market_update | market daily; market daily update; morning market wrap; market wrap; weekly market; weekly market intelligence; weekly market research; weekly market performance; market snapshot; market chartbook; india equity research daily; india equity research first call daily; india equity research first call daily report; technical market summary; equity market daily technical
technical_report | technical; technical update; technical stock calls; sectoral technical snapshot; elliott wave theory; quantitative outlook; quantitative market insights; market quantitative insights; index outlook
multi_company_sector | sectoral trends; sectoral updates; sectoral analysis; sectoral outlook; sector analysis; sector review; sector performance; industry trends; industry tidings; industry update; industry updates; multi sectoral; multiple sectors; sectoral research; sector deep dives; industry landscape; industry intelligence; sectoral statistics; multi sectoral industry tidings; thematic; structural themes; high growth sectors; sectoral inflection; sectoral premium; multi thematic weekly reading digest
fund_flow_report | flows; macro flows; institutional flows; fii sectoral flows; ownership trends; institutional positioning; equity ownership; broad market equity delivery positions; amfi semi annual categorization predictions; index rebalancing; indexing methodology
ipo_note | ipo flash; ipo anchor investor allocation; pre listing shareholder lock in expiry; ipo lock in analysis; price band revision
earnings_review | earnings; q1fy27 earnings; q1 fy27 earnings; 4qfy26 earnings review; multi sectoral earnings; q4fy26 earnings summary; 4qfy26 earnings trend check; q2fy26 earnings inflections; multi sectoral q1 fy27 performance; multi sectoral q4 fy26 performance; monthly sales review
earnings_preview | q1fy27 preview
pm_e_drive | ev policy; pm e bus sewa; one nation one charger
ev_policy_manufacturing | ev policy 2 0; delhi electric vehicle policy; draft delhi electric vehicle policy 2026 2030; vehicle scrappage policy; end of life vehicle rules; delhi ncr vehicle replacement scheme
cafe_norms | bs6; bs7; bs cev; cev stage v; euro vii; euro 7; cafe ii; cafe 3; cafe3 norms; cafe 3 norms; cpcb4; ie3 efficiency norms; cems compliance; epa 2027
almm | almm list iii; almm ii; almm mandate; almm other
ecms | mobile phone manufacturing scheme; ecms pli; emc 2 0; national policy on electronics 2019; national policy on electronics
india_semiconductor_mission | india semiconductor fabs scheme; indian semicon mission 2 0; india's semiconductor mission; semicon 1 0; government of odisha semiconductor policy
indiaai_mission | ai mission; bhashini; bhashni
national_electricity_plan | national electricity policy 2026; national generation adequacy plan; cea resource adequacy guidelines; cea resource adequacy plan; long term transmission plans; cea 900gw non fossil roadmap; ists waiver; gna fourth amendment; green energy open access rules 2022; electricity rights of consumers amendment rules 2026; power system development fund; psdf
electricity_act | electricity amendment act; ministry of power guidelines; uday; system improvement scheme
national_solar_mission | fdre; sjvn fdre scheme; national wind solar hybrid policy; seci; one sun one world one grid; pradhan mantri suryodaya yojana
viability_gap_funding | vgf scheme; vgf tranche ii
shanti_act | atomic energy act; nuclear energy mission; pfbr program; mahi banswara project
national_green_hydrogen_mission | green hydrogen policy; hydrogen economy roadmap
ethanol_blending_programme | e20 fuel blending mandate; e20 blending programme; ethanol blended petrol programme; ethanol blended petrol; gobardhan; gobardhan programme; gobardhan scheme; govardhan scheme; assam compressed biogas cbg policy 2026
epr | epr framework; epr norms; eu epr; pwmr; bwmr; hazardous and other wastes rules 2016; solid waste management rules 2016; swachh bharat mission; swachh maharashtra mission; simpler recycling legislation; non ferrous metal scrap recycling framework; recycling of ships act 2019; right to repair framework
trade_remedies | bis qco; stqc; stqc norms; stqc guidelines; quality control amendment order 2026; hallmarking; bis hallmarking; mandatory hallmarking; bureau of indian standards hallmarking; minimum price import; mip on pvc; mip for suspension grade pvc resin; bis er 01; iot system certification scheme; peso
foreign_trade_policy | advance authorisation scheme; market access initiative; national trade facilitation action plan 3 0; tonnage tax scheme; coastal cargo promotion scheme
shipbuilding_scheme | shipbuilding financial assistance policy; shipbuilding assistance scheme; ship breaking credit note scheme; shipbuilding and repair policy 2026 gujarat; harit nauka; parivartan scheme; parivartan fleet modernisation scheme
fta_other | cptpp; cafta dr; cepa; comprehensive economic partnership agreement; safta; asean; afcfta; uk fta; uk and eu fta; eu uk fta; india israel bilateral investment agreement; agreements on reciprocal trade; india australia strategic partnership
us_tariff_actions | section 122; ieepa; executive order 14409; executive order 14347; executive order 14411
us_legislation | american a i sovereign wealth fund act; one big beautiful bill; dodd frank banking reform; ndaa; national defense authorization act; stock act; sarbanes oxley; erisa; affordable care act; false claims act us; foreign account tax compliance act; clarity act; ai agent act; aim act; invest act; deal act; save act; speed act; uflpa; foreign investment risk review modernization act
eu_regulation | mica; gdpr; emir; mifid ii; mifir; aifmd; sfdr; eu taxonomy; reach; crd vi; psd2; eidas 2 0; digital markets act; eu battery regulation; critical raw materials act; eu critical raw materials act; digital product passports; eu digital product passport; nature restoration regulation
china_policy | anti involution; dual carbon policy; east data west compute; eastern data western computing; pboc credit repair policy; 14th five year cultural development plan; double reduction policy; china emergency robot development plan; guiding opinion on humanoid innovative development
remp_scheme | repm scheme; mmdr act; mines and minerals development and regulation act 1957; mines and minerals amendment bill 2026; mmdr amendment act 2026
fema | foreign exchange management act; foreign exchange management regulations; foreign exchange management non debt instruments rules 2019; fema remittance of assets regulations; rbi lrs; debt far; overseas portfolio investment; usd inr swap
fcnr_deposits | fcnr
ibc | cirp; corporate insolvency resolution process; ibc 2016; pmla
finance_act | income tax bill 2025; income tax rules 2026; income tax cuts; ltcg; cgt; section 80g; section 54f; vivad se vishwas ii; new tax amendment bill 2026; g sec tax exemption ordinance
gst | gst rate change; gst overhaul 2025; gst exemption on health insurance
rbi_regulations | ecl norms; lcr nsfr framework; lcr; nsfr; net stable funding ratio; scale based regulation; rbi psl framework; rbi nbfc ul framework; rbi microfinance regulatory framework; rbi credit card guidelines; rbi digital lending guidelines; rbi lending norms; rbi capital market exposure norms; rbi proprietary funding curbs; irac norms; mclr pricing policy; rbi special swap window; gold loan guidelines; banking laws amendment act 2025; nhb regulatory norms; dicgc
sebi_regulations | sebi f and o regulations; sebi research analyst regulations 2014; regulation 24a; regulation 33; securities contracts regulation act; scra; sebi anchor investor lock in regulations; t 1
sebi_fund_regulations | sebi mutual fund categorization; sebi portfolio managers regulations 2020; mf lite framework; ter rationalization; sebi expense ratio framework; sebi reit reclassification 2025; amfi semi annual categorization
sebi_derivatives | sebi index derivatives reforms 2024; sebi index equity options regulations
insurance_reforms | irdai bima bharosa; irdai 1 n mandate; insurance for all 2047; sabka bima sabki raksha act; bima trinity; insurance act; expense of management regulations
upi | mdr on upi; upi monetisation
eclgs | ecl scheme; eclgs 5; eclgs scheme 5 0; cgsmfi 2 0
pm_vishwakarma | raising and accelerating msme performance; skill india initiative; pmkvy
agri_schemes | pmfby; pmksy; pradhan mantri krishi sinchayee yojana; micro irrigation fund; national mission on edible oils oil palm; nmeo op; national mission on edible oilseeds and pulses scheme; mission for aatmanirbharta in pulses; pulses mission; agristack; digital crop survey; krishonnati yojana; kisan credit card; ahidf
pm_kusum | mukhyamantri saur krishi vahini yojana
ayushman_bharat | aarogya maitri
pmay | pmay urban 2
nep_2020 | nep ii; ugc online degree regulations; rte act 2009; parakh; atal tinkering labs; public examinations bill
labour_codes | code on social security 2020; social security code; occupational safety health and working conditions code 2020; apprentices act 1961
digital_india | uidai aadhaar project; aadhaar; digilocker; national single window system
state_industrial_policies | ts ipass; bihar industrial investment promotion policy 2016; new industrial and business development policy 2026; hyderabad industrial lands transformation policy; global capability centres policy 2026 haryana; gujarat gcc policy 2025 30; it policy 2023 jharkhand; odisha ict policy 2014; viksit up 2047; maharashtra green integrated data center park
make_in_india | create in india
carbon_credit_trading | paris agreement crediting mechanism; gx ets; ets; corsia; clean development mechanism; carbon removals certification framework; climate transition bond guidelines
"""

# Generic trailing role words stripped before matching (so "Pumps Manufacturers" -> "pumps").
ROLE_WORDS = """manufacturers manufacturer manufacturing makers maker producers producer processors
processor suppliers supplier exporters exporter operators operator providers provider companies
company players player developers firms products services other overall sector industry segment"""

# Ordered regex rules for long tails, applied ONLY when no alias matches.
# origin prefix | regex on the normalised text | tag_type | slug.  First match wins.
RULES = r"""
cls_ | \betfs?\b | subsector | etfs_funds
cls_ | \bfunds?\b | sector | amc
research_phrase | preview | doc_type | earnings_preview
research_phrase | \b[1-4]q ?fy ?\d\d\b|\bq[1-4] ?fy ?\d\d\b | doc_type | earnings_review
research_phrase | \bipos?\b | theme | primary_market_activity
policy_string | \bsebi\b | policy | sebi_regulations
policy_string | \brbi\b|reserve bank | policy | rbi_regulations
policy_string | \birdai\b|\binsurance\b|\bbima\b | policy | insurance_reforms
policy_string | \bgst\b | policy | gst
policy_string | \bpli\b|performance linked incentive | policy | pli_scheme_policy
policy_string | \bfta\b|free trade|trade agreement|partnership agreement | policy | fta_other
policy_string | executive order|house resolution|\bs \d{3,4}\b|\bus\b|american|\bu s\b|federal | policy | us_legislation
policy_string | \beu\b|european|\beuro\b | policy | eu_regulation
policy_string | china|chinese|pboc | policy | china_policy
policy_string | saudi|japan|korea|singapore|france|german|\buk\b|british|indonesia|malaysia|turkey|morocco|fiji|australia|brazil|argentin|mexic | policy | foreign_policy_other
policy_string | semicon | policy | india_semiconductor_mission
policy_string | \bev\b|electric vehicle|e bus|charger | policy | ev_policy_manufacturing
policy_string | solar|\bseci\b|renewable | policy | national_solar_mission
policy_string | hydrogen | policy | national_green_hydrogen_mission
policy_string | nuclear|atomic | policy | shanti_act
policy_string | electricity|\bcea\b|\bpower\b|discom | policy | national_electricity_plan
policy_string | \bcoal\b|lignite | policy | coal_policy
policy_string | \bgas\b|\bpng\b|\blng\b | policy | gas_policy
policy_string | defence|defense|military|\bdrdo\b|police|security | policy | defence_procurement
policy_string | telecom|spectrum|broadband | policy | telecom_policy
policy_string | \bai\b|artificial intelligence | policy | ai_policy_global
policy_string | \btax|income tax | policy | finance_act
policy_string | fertili|urea | policy | fertiliser_subsidy_policy
policy_string | krishi|kisan|\bagri|farm|crop|fisheries|matsya|dairy|livestock | policy | agri_schemes
policy_string | health|medical|\bdrug|pharma|ayush|aushadhi|cancer|blindness|immuni|nutrition|poshan | policy | health_policy
policy_string | educat|school|\bugc\b|skill | policy | nep_2020
policy_string | textile|weaver|handloom | policy | textile_schemes
policy_string | ship|maritime|\bport|coastal|cargo | policy | shipbuilding_scheme
policy_string | \brail|railway | policy | railway_policies
policy_string | \broad|highway|gram sadak | policy | rural_infra_schemes
policy_string | housing|urban|\bcity\b|master plan|municipal|co operative societies | policy | real_estate_regulation
policy_string | water|river|\bjal\b|delta | policy | water_resources_policy
policy_string | waste|recycl|plastic | policy | epr
policy_string | carbon|climate|emission|green|biodiversity | policy | carbon_credit_trading
policy_string | mines|mineral|mining | policy | remp_scheme
policy_string | msme|udyam|startup | policy | pm_vishwakarma
policy_string | pension|pay commission | policy | pay_commission_pension
policy_string | labour|labor|wage|employment|workforce | policy | labour_codes
policy_string | space|satellite | policy | space_programmes
policy_string | tourism|tourist | policy | tourism_policy
policy_string | excise|liquor | policy | state_excise_policy
policy_string | payment|\bupi\b | policy | payments_policy
policy_string | data protection|privacy|cyber|\bit act\b|information technology | policy | it_cyber_law
policy_string | gujarat|maharashtra|karnataka|tamil nadu|kerala|odisha|bihar|haryana|punjab|rajasthan|telangana|andhra|assam|delhi|jharkhand|uttar pradesh|\bup\b|madhya pradesh|west bengal|mumbai|mcgm | policy | state_industrial_policies
source_string | securities|broking|stock brokers?|investmart|capital markets?|finserve|financial services|shares? and | source | broker_other
source_string | asset manage|mutual fund|\bamc\b|capital management|investment manag|\baif\b|\bpms\b|wealth | source | pms_aif_house
source_string | universit|institute|foundation|centre for|center for|research service|journal|publishing|\bpress\b|academ | source | think_tank_academic
source_string | central bank|federal reserve|bundesbank|banque de|ministry|government|commission|authority|parliament|tribunal|regulator|\bsec\b | source | foreign_govt_regulator
source_string | \bbank\b | source | foreign_broker_other
source_string | consult|advisors?|advisory | source | consulting_firm
source_string | association|federation|council|summit|conference|forum|alliance | source | industry_association
source_string | times|news|magazine|daily|\bpost\b|digest|media|podcast|medium|newsletter | source | media_other
source_string | limited|\bltd\b|\binc\b|corporation|\bcorp\b|\bplc\b|\bllp\b|\bag\b|holdings|laboratories | source | company_self
source_string | . | source | independent_analyst
"""
