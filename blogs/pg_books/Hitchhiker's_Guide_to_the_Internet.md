---
layout: default
title: "Hitchhiker's Guide to the Internet"
parent: PG Books
has_children: false
---


<style>
.image-gallery {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  margin-bottom: 20px;
}

.image-row {
  display: flex;
  justify-content: flex-start;
  width: 100%;
  margin-bottom: 20px;
}

.image-item {
  width: 23%;
  margin-right: 2%;
  text-align: center;
}

.image-item:last-child {
  margin-right: 0;
}

.image-item img {
  width: 100%;
  height: auto;
  object-fit: cover;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.image-item p {
  margin-top: 5px;
  font-size: 0.9em;
  color: #555;
}

.video-container {
  margin: 20px 0;
}

.book-content {
  max-height: 500px;
  overflow-y: auto;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 5px;
  background-color: #f9f9f9;
  font-family: monospace;
  white-space: pre-wrap;
  margin-top: 20px;
}
</style>


# Hitchhiker's Guide to the Internet

<h3>Characters</h3>
<div class="image-gallery">
<div class="image-row">
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/characters/000_ed_krol.0.png" alt="000_ed_krol.0">
    <p>000_ed_krol.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/characters/002_dave_clark.0.png" alt="002_dave_clark.0">
    <p>002_dave_clark.0</p>
  </div>
</div>
</div>

<h3>Chapters</h3>
<div class="image-gallery">
<div class="image-row">
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/000_introduction.0.png" alt="000_introduction.0">
    <p>000_introduction.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/001_operating_the_internet.0.png" alt="001_operating_the_internet.0">
    <p>001_operating_the_internet.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/002_rfcs.0.png" alt="002_rfcs.0">
    <p>002_rfcs.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/004_nsfnet_network_service_center.0.png" alt="004_nsfnet_network_service_center.0">
    <p>004_nsfnet_network_service_center.0</p>
  </div>
</div>
<div class="image-row">
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/007_internet_problems.0.png" alt="007_internet_problems.0">
    <p>007_internet_problems.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/008_routing.0.png" alt="008_routing.0">
    <p>008_routing.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/009_names.0.png" alt="009_names.0">
    <p>009_names.0</p>
  </div>
  <div class="image-item">
    <img src="../results/Hitchhiker's Guide to the Internet/chapters/010_whats_wrong_with_berkeley_unix.0.png" alt="010_whats_wrong_with_berkeley_unix.0">
    <p>010_whats_wrong_with_berkeley_unix.0</p>
  </div>
</div>
</div>

<h2>Book Video</h2>
<div class="video-container">
  <video controls width="100%">
    <source src="../videos/Hitchhiker's Guide to the Internet.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Book Content

<div class="book-content">
﻿The Project Gutenberg eBook of Hitchhiker's Guide to the Internet
    
This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this ebook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

*** This is a COPYRIGHTED Project Gutenberg eBook. Details Below. ***
***     Please follow the copyright guidelines in this file.      ***


Title: Hitchhiker's Guide to the Internet

Author: Ed Krol

Release date: September 1, 1992 [eBook #39]
                Most recently updated: December 17, 2011

Language: English



*** START OF THE PROJECT GUTENBERG EBOOK HITCHHIKER'S GUIDE TO THE INTERNET ***














            The Hitchhikers Guide to the Internet
                        25 August 1987



                           Ed Krol
                    krol@uxc.cso.uiuc.edu




 This document was produced through funding of the National
 Science Foundation.

 Copyright (C) 1987, by the Board of Trustees of The University
 of Illinois.  Permission to duplicate this document, in whole
 or part, is granted provided reference is made to the source
 and this copyright is included in whole copies.


 This document assumes that one is familiar with the workings
 of a non-connected simple IP network (e.g. a few 4.2 BSD
 systems on an Ethernet not connected to anywhere else).
 Appendix A contains remedial information to get one to this
 point.  Its purpose is to get that person, familiar with a
 simple net, versed in the "oral tradition" of the Internet
 to the point that that net can be connected to the Internet
 with little danger to either.  It is not a tutorial, it
 consists of pointers to other places, literature, and hints
 which are not normally documented.  Since the Internet is a
 dynamic environment, changes to this document will be made
 regularly.  The author welcomes comments and suggestions.
 This is especially true of terms for the glossary (definitions
 are not necessary).




 In the beginning there was the ARPAnet, a wide area
 experimental network connecting hosts and terminal servers
 together.  Procedures were set up to regulate the allocation
 of addresses and to create voluntary standards for the network.
 As local area networks became more pervasive, many hosts became
 gateways to local networks.  A network layer to allow the
 interoperation of these networks was developed and called IP
 (Internet Protocol).  Over time other groups created long haul
 IP based networks (NASA, NSF, states...).  These nets, too,
 interoperate because of IP.  The collection of all of these
 interoperating networks is the Internet.

 Two groups do much of the research and information work of
 the Internet (ISI and SRI).  ISI (the Informational Sciences
 Institute) does much of the research, standardization, and
 allocation work of the Internet.  SRI International provides
 information services for the Internet.  In fact, after you
 are connected to the Internet most of the information in
 this document can be retrieved from the Network Information
 Center (NIC) run by SRI.



 Operating the Internet

 Each network, be it the ARPAnet, NSFnet or a regional network,
 has its own operations center.  The ARPAnet is run by
 BBN, Inc. under contract from DARPA.  Their facility is
 called the Network Operations Center or NOC.  Cornell
 University temporarily operates NSFnet (called the Network
 Information Service Center, NISC).  It goes on to the

                             -2-

 regionals having similar facilities to monitor and keep
 watch over the goings on of their portion of the Internet.
 In addition, they all should have some knowledge of what is
 happening to the Internet in total. If a problem comes up,
 it is suggested that a campus network liaison should contact
 the network operator to which he is directly connected. That
 is, if you are connected to a regional network (which is
 gatewayed to the NSFnet, which is connected to the
 ARPAnet...)  and have a problem, you should contact your
 regional network operations center.


 RFCs

 The internal workings of the Internet are defined by a set
 of documents called RFCs (Request for Comments).  The general
 process for creating an RFC is for someone wanting something
 formalized to write a document describing the issue and mailing
 it to Jon Postel (postel@isi.edu).  He acts as a referee for
 the proposal.  It is then commented upon by all those wishing
 to take part in the discussion (electronically of course).
 It may go through multiple revisions.  Should it be generally
 accepted as a good idea, it will be assigned a number and
 filed with the RFCs.

 The RFCs can be divided into five groups: required, suggested,
 directional, informational and obsolete.  Required RFC's (e.g.
 RFC-791, The Internet Protocol) must be implemented on any host
 connected to the Internet.  Suggested RFCs are generally
 implemented by network hosts.  Lack of them does not preclude
 access to the Internet, but may impact its usability.  RFC-793
 (Transmission Control Protocol) is a suggested RFC.  Directional
 RFCs were discussed and agreed to, but their application has never
 come into wide use.  This may be due to the lack of wide need for
 the specific application (RFC-937 The Post Office Protocol) or
 that, although technically superior, ran against other pervasive
 approaches (RFC-891 Hello).  It is suggested that should the
 facility be required by a particular site, animplementation
 be done in accordance with the RFC.  This insures that, should
 the idea be one whose time has come, the implementation will be
 in accordance with some standard and will be generally usable.
 Informational RFCs contain factual information about the
 Internet and its operation (RFC-990, Assigned Numbers).
 Finally, as the Internet and technology have grown, some
 RFCs have become unnecessary.  These obsolete RFCs cannot
 be ignored, however.  Frequently when a change is made to
 some RFC that causes a new one to be issued obsoleting others,
 the new RFC only contains explanations and motivations for the
 change.  Understanding the model on which the whole facility
 is based may involve reading the original and subsequent RFCs
 on the topic.

                             -3-

 (Appendix B contains a list of what are considered to be the
 major RFCs necessary for understanding the Internet).



 The Network Information Center

 The NIC is a facility available to all Internet users which
 provides information to the community.  There are three
 means of NIC contact: network, telephone, and mail.  The
 network accesses are the most prevalent.  Interactive access
 is frequently used to do queries of NIC service overviews,
 look up user and host names, and scan lists of NIC documents.
 It is available by using

      %telnet sri-nic.arpa

 on a BSD system and following the directions provided by a
 user friendly prompter.  From poking around in the databases
 provided one might decide that a document named NETINFO:NUG.DOC
 (The Users Guide to the ARPAnet) would be worth having.  It could
 be retrieved via an anonymous FTP.  An anonymous FTP would proceed
 something like the following.  (The dialogue may vary slightly
 depending on the implementation of FTP you are using).

      %ftp sri-nic.arpa
      Connected to sri-nic.arpa.
      220 SRI_NIC.ARPA FTP Server Process 5Z(47)-6 at Wed
17-Jun-87 12:00 PDT
      Name (sri-nic.arpa:myname): anonymous
      331 ANONYMOUS user ok, send real ident as password.
      Password: myname
      230 User ANONYMOUS logged in at Wed 17-Jun-87 12:01 PDT,
job 15.
      ftp> get netinfo:nug.doc
      200 Port 18.144 at host 128.174.5.50 accepted.
      150 ASCII retrieve of <NETINFO>NUG.DOC.11 started.
      226 Transfer Completed 157675 (8) bytes transferred
      local: netinfo:nug.doc  remote:netinfo:nug.doc
      157675 bytes in 4.5e+02 seconds (0.34 Kbytes/s)
      ftp> quit
      221 QUIT command received. Goodbye.

 (Another good initial document to fetch is
 NETINFO:WHAT-THE-NIC-DOES.TXT)!

 Questions of the NIC or problems with services can be asked
 of or reported to using electronic mail.  The following
 addresses can be used:

      NIC@SRI-NIC.ARPA         General user assistance, document requests
      REGISTRAR@SRI-NIC.ARPA   User registration and WHOIS updates
      HOSTMASTER@SRI-NIC.ARPA  Hostname and domain changes and updates
      ACTION@SRI-NIC.ARPA      SRI-NIC computer operations
      SUGGESTIONS@SRI-NIC.ARPA Comments on NIC publications and services

                             -4-

 For people without network access, or if the number of documents
 is large, many of the NIC documents are available in printed
 form for a small charge.  One frequently ordered document for
 starting sites is a compendium of major RFCs.  Telephone access is
 used primarily for questions or problems with network access.
 (See appendix B for mail/telephone contact numbers).



 The NSFnet Network Service Center

 The NSFnet Network Service Center (NNSC) is funded by NSF to
 provide a first level of aid to users of NSFnet should they
 have questions or encounter problems traversing the network.
 It is run by BBN Inc.  Karen Roubicek
 (roubicek@nnsc.nsf.net) is the NNSC user liaison.

 The NNSC, which currently has information and documents
 online and in printed form, plans to distribute news through
 network mailing lists, bulletins, newsletters, and online
 reports.  The NNSC also maintains a database of contact
 points and sources of additional information about NSFnet
 component networks and supercomputer centers.

 Prospective or current users who do not know whom to call
 concerning questions about NSFnet use, should contact the
 NNSC.  The NNSC ...

[Content truncated for display]
</div>
